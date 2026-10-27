import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_conv_softmax_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    total, C_IN, D, H, W, C_OUT, OD, OH, OW,
    KS: tl.constexpr, P: tl.constexpr,
    BLOCK: tl.constexpr, COUT_P: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    ow = offs % OW
    t = offs // OW
    oh = t % OH
    t = t // OH
    od = t % OD
    n = t // OD

    co = tl.arange(0, COUT_P)
    cmask = co < C_OUT
    bias = tl.load(b_ptr + co, mask=cmask, other=0.0).to(tl.float32)

    HW = H * W
    res = tl.full([BLOCK, COUT_P], float("-inf"), tl.float32)

    for p in range(P * P * P):
        pd = p // (P * P)
        ph = (p // P) % P
        pw = p % P
        d0 = od * P + pd
        h0 = oh * P + ph
        w0 = ow * P + pw

        acc = tl.zeros([BLOCK, COUT_P], tl.float32) + bias[None, :]
        for ci in range(C_IN):
            x_base = (n * C_IN + ci) * D * HW
            w_base = (co * C_IN + ci) * (KS * KS * KS)
            for kd in tl.static_range(KS):
                for kh in tl.static_range(KS):
                    for kw in tl.static_range(KS):
                        xv = tl.load(
                            x_ptr + x_base + (d0 + kd) * HW + (h0 + kh) * W + (w0 + kw),
                            mask=mask, other=0.0,
                        ).to(tl.float32)
                        wv = tl.load(
                            w_ptr + w_base + (kd * KS + kh) * KS + kw,
                            mask=cmask, other=0.0,
                        ).to(tl.float32)
                        acc += xv[:, None] * wv[None, :]

        acc = tl.where(cmask[None, :], acc, float("-inf"))
        m = tl.max(acc, axis=1)
        e = tl.exp(acc - m[:, None])
        e = tl.where(cmask[None, :], e, 0.0)
        s = tl.sum(e, axis=1)
        sm = e / s[:, None]
        res = tl.maximum(res, sm)

    OHW = OH * OW
    out_off = ((n[:, None] * C_OUT + co[None, :]) * OD + od[:, None]) * OHW \
        + oh[:, None] * OW + ow[:, None]
    tl.store(out_ptr + out_off, res.to(out_ptr.dtype.element_ty),
             mask=mask[:, None] & cmask[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool1 = nn.MaxPool3d(pool_kernel_size)
        self.pool2 = nn.MaxPool3d(pool_kernel_size)
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, C_IN, D, H, W = x.shape
        KS = self.kernel_size
        P = self.pool_kernel_size
        C_OUT = self.conv.out_channels
        CD, CH, CW = D - KS + 1, H - KS + 1, W - KS + 1
        OD, OH, OW = (CD // P) // P, (CH // P) // P, (CW // P) // P
        PP = P * P

        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        out = torch.empty((N, C_OUT, OD, OH, OW), device=x.device, dtype=x.dtype)
        total = N * OD * OH * OW
        if total == 0:
            return out
        BLOCK = 64
        COUT_P = triton.next_power_of_2(C_OUT)
        grid = (triton.cdiv(total, BLOCK),)
        fused_conv_softmax_pool_kernel[grid](
            x, w, b, out,
            total, C_IN, D, H, W, C_OUT, OD, OH, OW,
            KS=KS, P=PP, BLOCK=BLOCK, COUT_P=COUT_P,
        )
        return out
