import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_sub_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    C, H, W, OC, OH, OW,
    sub1, sub2,
    KS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_p = tl.program_id(0)
    n = tl.program_id(1).to(tl.int64)

    OHW = OH * OW
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    mask_p = offs_p < OHW
    oh = offs_p // OW
    ow = offs_p % OW

    offs_oc = tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    acc = tl.zeros((BLOCK_P, BLOCK_OC), dtype=tl.float32)

    x_base = x_ptr + n * C * H * W
    KK: tl.constexpr = KS * KS
    for c in range(0, C):
        x_c = x_base + c * H * W
        for kh in tl.static_range(KS):
            for kw in tl.static_range(KS):
                ih = oh + kh
                iw = ow + kw
                xv = tl.load(x_c + ih * W + iw, mask=mask_p, other=0.0).to(tl.float32)
                wv = tl.load(w_ptr + offs_oc * (C * KK) + c * KK + kh * KS + kw,
                             mask=mask_oc, other=0.0).to(tl.float32)
                acc += xv[:, None] * wv[None, :]

    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0).to(tl.float32)
    v = acc + bias[None, :]
    v = v - sub1
    v = v - sub2

    # mish: v * tanh(softplus(v))
    sp = tl.maximum(v, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(v)))
    e = tl.exp(-2.0 * sp)
    t = (1.0 - e) / (1.0 + e)
    res = v * t

    out_base = out_ptr + n * OC * OHW
    out_ptrs = out_base + offs_oc[None, :] * OHW + offs_p[:, None]
    tl.store(out_ptrs, res.to(out_ptr.dtype.element_ty),
             mask=mask_p[:, None] & mask_oc[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        N, C, H, W = x.shape
        OC = w.shape[0]
        KS = w.shape[2]
        OH = H - KS + 1
        OW = W - KS + 1
        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)
        BLOCK_P = 128
        BLOCK_OC = triton.next_power_of_2(OC)
        grid = (triton.cdiv(OH * OW, BLOCK_P), N)
        conv_sub_mish_kernel[grid](
            x, w, b, out,
            C, H, W, OC, OH, OW,
            float(self.subtract_value_1), float(self.subtract_value_2),
            KS=KS, BLOCK_P=BLOCK_P, BLOCK_OC=BLOCK_OC,
            num_warps=4,
        )
        return out
