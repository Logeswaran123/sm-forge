import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_hswish_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, Cout, H, W, OH, OW,
    C: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < Cout

    OHW = OH * OW
    n = offs_m // OHW
    rem = offs_m % OHW
    oh = rem // OW
    ow = rem % OW

    x_base = n * (C * H * W) + oh * W + ow
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for c in tl.static_range(C):
        for kh in tl.static_range(K):
            for kw in tl.static_range(K):
                xv = tl.load(x_ptr + x_base + c * H * W + kh * W + kw,
                             mask=mask_m, other=0.0).to(tl.float32)
                wv = tl.load(w_ptr + offs_n * (C * K * K) + c * K * K + kh * K + kw,
                             mask=mask_n, other=0.0).to(tl.float32)
                acc += xv[:, None] * wv[None, :]

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    # hardswish
    t = tl.minimum(tl.maximum(acc + 3.0, 0.0), 6.0)
    acc = acc * t / 6.0
    # relu
    acc = tl.maximum(acc, 0.0)

    out_off = (n * (Cout * OHW) + rem)[:, None] + (offs_n * OHW)[None, :]
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, acc.to(out_ptr.dtype.element_ty), mask=mask)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        N, C, H, W = x.shape
        Cout = w.shape[0]
        K = w.shape[2]
        OH = H - K + 1
        OW = W - K + 1
        out = torch.empty((N, Cout, OH, OW), device=x.device, dtype=x.dtype)
        M = N * OH * OW
        BLOCK_M = 128
        BLOCK_N = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(Cout, BLOCK_N))
        conv_hswish_relu_kernel[grid](
            x, w, b, out,
            M, Cout, H, W, OH, OW,
            C=C, K=K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out
