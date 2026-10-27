import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_bn_scale_kernel(
    x_ptr, w_ptr, b_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr, out_ptr,
    N, C_in, H, W, C_out, OH, OW,
    eps, scaling,
    HAS_BIAS: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    OHW = OH * OW
    M = N * OHW
    K = C_in * KH * KW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < C_out

    n_idx = offs_m // OHW
    rem = offs_m % OHW
    oh = rem // OW
    ow = rem % OW
    # base offset for each output pixel (input pixel at top-left of window)
    a_base = n_idx * (C_in * H * W) + oh * W + ow  # [BLOCK_M]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        ci = offs_k // (KH * KW)
        r = offs_k % (KH * KW)
        kh = r // KW
        kw = r % KW
        k_off = ci * (H * W) + kh * W + kw  # [BLOCK_K]

        a_ptrs = x_ptr + a_base[:, None] + k_off[None, :]
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # weight [C_out, K] contiguous -> B[k, co] = w[co*K + k]
        b_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc = tl.dot(a, b.to(a.dtype), acc)

    # epilogue: BN (eval) + scaling
    mean = tl.load(mean_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    else:
        bias = tl.zeros((BLOCK_N,), dtype=tl.float32)

    inv_std = 1.0 / tl.sqrt(var + eps)
    s = gamma * inv_std * scaling
    t = ((bias - mean) * gamma * inv_std + beta) * scaling
    y = acc * s[None, :] + t[None, :]

    out_off = n_idx[:, None] * (C_out * OHW) + offs_n[None, :] * OHW + rem[:, None]
    tl.store(out_ptr + out_off, y.to(out_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        N, C_in, H, W = x.shape
        C_out, _, KH, KW = w.shape
        OH = H - KH + 1
        OW = W - KW + 1
        out = torch.empty((N, C_out, OH, OW), device=x.device, dtype=x.dtype)

        has_bias = self.conv.bias is not None
        bias = self.conv.bias if has_bias else self.bn.running_mean

        BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 32
        M = N * OH * OW
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(C_out, BLOCK_N))
        conv_bn_scale_kernel[grid](
            x, w, bias,
            self.bn.running_mean, self.bn.running_var,
            self.bn.weight, self.bn.bias, out,
            N, C_in, H, W, C_out, OH, OW,
            float(self.bn.eps), float(self.scaling_factor),
            HAS_BIAS=has_bias, KH=KH, KW=KW,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )
        return out
