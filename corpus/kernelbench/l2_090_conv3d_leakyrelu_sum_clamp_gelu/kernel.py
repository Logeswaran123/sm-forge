import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, b_ptr, s_ptr, out_ptr,
    M, Cout, K,
    Cin, D, H, W,
    Do, Ho, Wo,
    HAS_BIAS: tl.constexpr,
    KS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < Cout

    # decode output position
    ow = offs_m % Wo
    t = offs_m // Wo
    oh = t % Ho
    t = t // Ho
    od = t % Do
    nb = t // Do

    sH = W
    sD = H * W
    sC = D * H * W
    sN = Cin * sC

    x_base = nb * sN + od * sD + oh * sH + ow  # int64 [BLOCK_M]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KK: tl.constexpr = KS * KS * KS
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        ci = offs_k // KK
        r = offs_k % KK
        kd = r // (KS * KS)
        r2 = r % (KS * KS)
        kh = r2 // KS
        kw = r2 % KS
        k_off = (ci * sC + kd * sD + kh * sH + kw).to(tl.int64)

        a_ptrs = x_ptr + x_base[:, None] + k_off[None, :]
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        w_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]
        wt = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        wt = wt.to(a.dtype)

        acc += tl.dot(a, wt)

    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc += bias[None, :]

    # leaky relu
    acc = tl.where(acc >= 0, acc, acc * 0.2)
    # add sum tensor (per channel)
    s = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc + s[None, :]
    # clamp
    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)
    # exact gelu
    acc = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))

    S = Do * Ho * Wo
    sp = od * (Ho * Wo) + oh * Wo + ow
    out_row = nb * Cout * S + sp
    out_ptrs = out_ptr + out_row[:, None] + offs_n[None, :].to(tl.int64) * S
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        Cout, Cin, KS, _, _ = w.shape
        N, _, D, H, W = x.shape
        Do, Ho, Wo = D - KS + 1, H - KS + 1, W - KS + 1
        M = N * Do * Ho * Wo
        K = Cin * KS * KS * KS

        out = torch.empty((N, Cout, Do, Ho, Wo), device=x.device, dtype=x.dtype)
        w2 = w.view(Cout, K)
        bias = self.conv.bias
        has_bias = bias is not None
        b = bias.contiguous() if has_bias else w2
        s = self.sum_tensor.contiguous().view(-1)

        BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(Cout, BLOCK_N))
        conv3d_fused_kernel[grid](
            x, w2, b, s, out,
            M, Cout, K,
            Cin, D, H, W,
            Do, Ho, Wo,
            HAS_BIAS=has_bias, KS=KS,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )
        return out
