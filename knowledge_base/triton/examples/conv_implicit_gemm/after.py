import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_hswish_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    NB, C, H, W, OC, OH, OW, K,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    M = NB * OH * OW
    mask_m = offs_m < M
    mask_n = offs_n < OC

    ohw = OH * OW
    n_idx = offs_m // ohw
    rem = offs_m % ohw
    oh = rem // OW
    ow = rem % OW
    # base address of top-left input element for each output pixel
    x_base = n_idx * (C * H * W) + oh * W + ow  # [BLOCK_M]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        c = offs_k // (KH * KW)
        r = offs_k % (KH * KW)
        kh = r // KW
        kw = r % KW
        k_off = c * (H * W) + kh * W + kw  # [BLOCK_K]

        a_ptrs = x_ptr + x_base[:, None] + k_off[None, :]
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        w_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]
        wv = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        wv = wv.to(a.dtype)

        acc += tl.dot(a, wv)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    gate = tl.minimum(tl.maximum((acc + 3.0) / 6.0, 0.0), 1.0)
    acc = acc * gate
    acc = tl.maximum(acc, 0.0)

    out_off = (n_idx * (OC * ohw) + oh * OW + ow)[:, None] + offs_n[None, :] * ohw
    tl.store(out_ptr + out_off, acc.to(out_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        NB, C, H, W = x.shape
        OC, _, KH, KW = w.shape
        OH = H - KH + 1
        OW = W - KW + 1
        K = C * KH * KW
        out = torch.empty((NB, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 16
        M = NB * OH * OW
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(OC, BLOCK_N))
        conv_hswish_relu_kernel[grid](
            x, w, b, out,
            NB, C, H, W, OC, OH, OW, K,
            KH=KH, KW=KW,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return out
