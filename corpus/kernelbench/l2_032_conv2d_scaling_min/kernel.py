import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_scale_min_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, C, H, W, OH, OW, COUT, K,
    scale,
    KS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    ohw = OH * OW
    n_idx = offs_m // ohw
    rem = offs_m % ohw
    oh = rem // OW
    ow = rem % OW
    # base address for each output pixel (top-left of receptive field, channel 0)
    x_base = n_idx.to(tl.int64) * C * H * W + oh * W + ow  # [BLOCK_M]

    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < COUT

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    HW = H * W
    KK: tl.constexpr = KS * KS

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        c = offs_k // KK
        r = offs_k % KK
        kh = r // KS
        kw = r % KS
        k_off = c * HW + kh * W + kw  # [BLOCK_K]

        a_ptrs = x_ptr + x_base[:, None] + k_off[None, :]
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        w_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]
        b = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc = tl.dot(a.to(tl.float16), b.to(tl.float16), acc)

    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    val = (acc + bias[None, :]) * scale
    val = tl.where(n_mask[None, :], val, float("inf"))
    mn = tl.min(val, axis=1)

    tl.store(out_ptr + offs_m, mn.to(out_ptr.dtype.element_ty), mask=m_mask)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = x.contiguous()
        N, C, H, W = x.shape
        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()
        COUT = weight.shape[0]
        KS = weight.shape[2]
        OH = H - KS + 1
        OW = W - KS + 1
        K = C * KS * KS
        M = N * OH * OW

        out = torch.empty((N, 1, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_M = 64
        BLOCK_N = max(16, triton.next_power_of_2(COUT))
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M),)
        conv_scale_min_kernel[grid](
            x, weight, bias, out,
            M, C, H, W, OH, OW, COUT, K,
            float(self.scale_factor),
            KS=KS,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )
        return out
