import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_div_lrelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_IN, H, W, C_OUT, OH, OW,
    divisor, neg_slope,
    HAS_BIAS: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_CO: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_co = tl.program_id(1)

    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)

    total_p = N * OH * OW
    mask_p = offs_p < total_p
    mask_co = offs_co < C_OUT

    ow = offs_p % OW
    tmp = offs_p // OW
    oh = tmp % OH
    n = tmp // OH

    x_base = n.to(tl.int64) * C_IN * H * W + (oh * W + ow).to(tl.int64)

    acc = tl.zeros((BLOCK_P, BLOCK_CO), dtype=tl.float32)
    for ci in range(0, C_IN):
        x_ci = x_base + ci * H * W
        w_ci = offs_co * (C_IN * KH * KW) + ci * (KH * KW)
        for kh in tl.static_range(KH):
            for kw in tl.static_range(KW):
                xv = tl.load(x_ptr + x_ci + (kh * W + kw), mask=mask_p, other=0.0).to(tl.float32)
                wv = tl.load(w_ptr + w_ci + (kh * KW + kw), mask=mask_co, other=0.0).to(tl.float32)
                acc += xv[:, None] * wv[None, :]

    if HAS_BIAS:
        bv = tl.load(b_ptr + offs_co, mask=mask_co, other=0.0).to(tl.float32)
        acc += bv[None, :]

    acc = acc / divisor
    acc = tl.where(acc >= 0, acc, acc * neg_slope)

    out_off = (n.to(tl.int64) * C_OUT * OH * OW + (oh * OW + ow).to(tl.int64))[:, None] \
        + (offs_co.to(tl.int64) * OH * OW)[None, :]
    mask = mask_p[:, None] & mask_co[None, :]
    tl.store(out_ptr + out_off, acc.to(out_ptr.dtype.element_ty), mask=mask)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias
        N, C_IN, H, W = x.shape
        C_OUT, _, KH, KW = w.shape
        OH = H - KH + 1
        OW = W - KW + 1
        out = torch.empty((N, C_OUT, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_P = 128
        BLOCK_CO = 64
        grid = (triton.cdiv(N * OH * OW, BLOCK_P), triton.cdiv(C_OUT, BLOCK_CO))
        has_bias = b is not None
        b_arg = b.contiguous() if has_bias else w
        conv_div_lrelu_kernel[grid](
            x, w, b_arg, out,
            N, C_IN, H, W, C_OUT, OH, OW,
            float(self.divisor), 0.01,
            HAS_BIAS=has_bias,
            KH=KH, KW=KW,
            BLOCK_P=BLOCK_P, BLOCK_CO=BLOCK_CO,
            num_warps=4,
        )
        return out
