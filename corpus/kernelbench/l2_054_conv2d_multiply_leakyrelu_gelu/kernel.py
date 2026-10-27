import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_mul_lrelu_gelu_kernel(
    x_ptr, w_ptr, b_ptr, mul_ptr, out_ptr,
    NB, C, H, W, OC, OH, OW, M, K,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    ow = offs_m % OW
    t = offs_m // OW
    oh = t % OH
    n = t // OH

    HW = H * W
    x_base = n.to(tl.int64) * C * HW + (oh * W + ow).to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        ic = offs_k // (KH * KW)
        r = offs_k % (KH * KW)
        kh = r // KW
        kw = r % KW
        k_off = (ic * HW + kh * W + kw).to(tl.int64)

        a = tl.load(x_ptr + x_base[:, None] + k_off[None, :],
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(w_ptr + offs_n[None, :] * K + offs_k[:, None],
                    mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, b.to(a.dtype), acc)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    mul = tl.load(mul_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    y = (acc + bias[None, :]) * mul[None, :]
    y = tl.where(y >= 0, y, y * 0.01)
    y = 0.5 * y * (1.0 + tl.erf(y * 0.7071067811865476))

    OHW = OH * OW
    out_off = (n.to(tl.int64) * OC * OHW + (oh * OW + ow).to(tl.int64))[:, None] \
        + (offs_n.to(tl.int64) * OHW)[None, :]
    tl.store(out_ptr + out_off, y.to(out_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = x.contiguous()
        NB, C, H, W = x.shape
        w = self.conv.weight.contiguous()
        OC, _, KH, KW = w.shape
        OH = H - KH + 1
        OW = W - KW + 1
        K = C * KH * KW
        M = NB * OH * OW
        w2 = w.view(OC, K)
        bias = self.conv.bias
        if bias is None:
            bias = torch.zeros(OC, device=x.device, dtype=torch.float32)
        bias = bias.contiguous()
        mul = self.multiplier.contiguous().view(-1)

        out = torch.empty((NB, OC, OH, OW), device=x.device, dtype=x.dtype)
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(OC, BLOCK_N))
        conv_mul_lrelu_gelu_kernel[grid](
            x, w2, bias, mul, out,
            NB, C, H, W, OC, OH, OW, M, K,
            KH=KH, KW=KW,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )
        return out
