import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_min_bias_scale_kernel(
    x_ptr, w_ptr, cb_ptr, b_ptr, out_ptr,
    NB, C, H, W, OC, OH, OW,
    sxn, sxc, sxh, sxw,
    swo, swc, swh, sww,
    son, soc, soh, sow,
    const_val, scale,
    HAS_CB: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = NB * OH * OW
    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < M
    offs_m_c = tl.where(mask_m, offs_m, 0)
    n = offs_m_c // (OH * OW)
    rem = offs_m_c % (OH * OW)
    oh = rem // OW
    ow = rem % OW
    n64 = n.to(tl.int64)

    offs_oc = pid_n * BN + tl.arange(0, BN)
    mask_oc = offs_oc < OC
    offs_oc_c = tl.where(mask_oc, offs_oc, 0)

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for kh in tl.static_range(KH):
        for kw in tl.static_range(KW):
            a_base = n64 * sxn + (oh + kh) * sxh + (ow + kw) * sxw  # [BM]
            b_base = offs_oc_c * swo + kh * swh + kw * sww           # [BN]
            for c0 in range(0, C, BK):
                cc = c0 + tl.arange(0, BK)
                mask_c = cc < C
                a_ptrs = x_ptr + a_base[:, None] + cc[None, :] * sxc
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_c[None, :], other=0.0)
                b_ptrs = w_ptr + b_base[None, :] + cc[:, None] * swc
                b = tl.load(b_ptrs, mask=mask_c[:, None] & mask_oc[None, :], other=0.0)
                acc += tl.dot(a.to(tl.float16), b.to(tl.float16))

    if HAS_CB:
        cb = tl.load(cb_ptr + offs_oc_c, mask=mask_oc, other=0.0).to(tl.float32)
        acc = acc + cb[None, :]
    acc = tl.minimum(acc, const_val)
    bb = tl.load(b_ptr + offs_oc_c, mask=mask_oc, other=0.0).to(tl.float32)
    acc = (acc + bb[None, :]) * scale

    o_ptrs = out_ptr + (n64 * son + oh * soh + ow * sow)[:, None] + offs_oc_c[None, :] * soc
    tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_oc[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        cb = self.conv.bias
        NB, C, H, W = x.shape
        OC, _, KH, KW = w.shape
        OH = H - KH + 1
        OW = W - KW + 1
        out = torch.empty((NB, OC, OH, OW), device=x.device, dtype=x.dtype)
        bias = self.bias.reshape(-1).contiguous()
        has_cb = cb is not None
        cb_t = cb.contiguous() if has_cb else bias

        BM, BN, BK = 128, 128, 32
        M = NB * OH * OW
        grid = (triton.cdiv(M, BM), triton.cdiv(OC, BN))
        conv_min_bias_scale_kernel[grid](
            x, w, cb_t, bias, out,
            NB, C, H, W, OC, OH, OW,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            float(self.constant_value), float(self.scaling_factor),
            HAS_CB=has_cb, KH=KH, KW=KW,
            BM=BM, BN=BN, BK=BK,
            num_warps=8, num_stages=2,
        )
        return out
