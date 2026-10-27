import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_convt_pool_softmax_swish_max_kernel(
    x_ptr, w_ptr, b_ptr, sub_ptr, out_ptr,
    TOTAL,
    C_IN, D_IN, H_IN, W_IN,
    D_CONV, H_CONV, W_CONV,
    PD, PH, PW,
    C_OUT: tl.constexpr, C_PAD: tl.constexpr,
    K: tl.constexpr, STRIDE: tl.constexpr, PAD: tl.constexpr,
    PK: tl.constexpr, PS: tl.constexpr, PP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL

    pw = offs % PW
    t0 = offs // PW
    ph = t0 % PH
    t0 = t0 // PH
    pd = t0 % PD
    n = t0 // PD

    co = tl.arange(0, C_PAD)
    cmask = co < C_OUT
    bias = tl.load(b_ptr + co, mask=cmask, other=0.0).to(tl.float32)
    subv = tl.load(sub_ptr + co, mask=cmask, other=0.0).to(tl.float32)

    KKK: tl.constexpr = K * K * K
    neg_inf = float("-inf")
    pooled = tl.full([BLOCK, C_PAD], neg_inf, dtype=tl.float32)

    for pkd in tl.static_range(PK):
        od = pd * PS - PP + pkd
        for pkh in tl.static_range(PK):
            oh = ph * PS - PP + pkh
            for pkw in tl.static_range(PK):
                ow = pw * PS - PP + pkw
                pos_valid = (od >= 0) & (od < D_CONV) & (oh >= 0) & (oh < H_CONV) & (ow >= 0) & (ow < W_CONV)
                acc = tl.zeros([BLOCK, C_PAD], dtype=tl.float32) + bias[None, :]
                for ci in range(0, C_IN):
                    x_base = (n * C_IN + ci) * D_IN
                    w_base = ci * C_OUT * KKK
                    for kd in tl.static_range(K):
                        td = od + PAD - kd
                        idd = td // STRIDE
                        vd = (td >= 0) & ((td % STRIDE) == 0) & (idd < D_IN)
                        for kh in tl.static_range(K):
                            th = oh + PAD - kh
                            ih = th // STRIDE
                            vh = (th >= 0) & ((th % STRIDE) == 0) & (ih < H_IN)
                            for kw in tl.static_range(K):
                                tw = ow + PAD - kw
                                iw = tw // STRIDE
                                vw = (tw >= 0) & ((tw % STRIDE) == 0) & (iw < W_IN)
                                valid = mask & pos_valid & vd & vh & vw
                                x_off = ((x_base + idd) * H_IN + ih) * W_IN + iw
                                xv = tl.load(x_ptr + x_off, mask=valid, other=0.0).to(tl.float32)
                                wv = tl.load(w_ptr + w_base + co * KKK + (kd * K + kh) * K + kw,
                                             mask=cmask, other=0.0).to(tl.float32)
                                acc += xv[:, None] * wv[None, :]
                acc = tl.where(pos_valid[:, None], acc, neg_inf)
                pooled = tl.maximum(pooled, acc)

    # softmax over channels
    pooled = tl.where(cmask[None, :], pooled, neg_inf)
    m = tl.max(pooled, axis=1)
    e = tl.exp(pooled - m[:, None])
    e = tl.where(cmask[None, :], e, 0.0)
    s = tl.sum(e, axis=1)
    sm = e / s[:, None]
    v = sm - subv[None, :]
    sw = v * (1.0 / (1.0 + tl.exp(-v)))
    sw = tl.where(cmask[None, :], sw, neg_inf)
    res = tl.max(sw, axis=1)
    tl.store(out_ptr + offs, res.to(out_ptr.dtype.element_ty), mask=mask)


def _triple(v):
    if isinstance(v, (tuple, list)):
        assert len(set(v)) == 1
        return int(v[0])
    return int(v)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding,
                 pool_kernel_size, pool_stride, pool_padding):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                                 padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = _triple(kernel_size)
        self.stride = _triple(stride)
        self.pad = _triple(padding)
        self.out_pad = _triple(output_padding)
        self.pk = _triple(pool_kernel_size)
        self.ps = _triple(pool_stride if pool_stride is not None else pool_kernel_size)
        self.pp = _triple(pool_padding)

    def forward(self, x):
        x = x.contiguous()
        N, C_in, D, H, W = x.shape
        k, s, p, op = self.k, self.stride, self.pad, self.out_pad
        Dc = (D - 1) * s - 2 * p + k + op
        Hc = (H - 1) * s - 2 * p + k + op
        Wc = (W - 1) * s - 2 * p + k + op
        PD = (Dc + 2 * self.pp - self.pk) // self.ps + 1
        PH = (Hc + 2 * self.pp - self.pk) // self.ps + 1
        PW = (Wc + 2 * self.pp - self.pk) // self.ps + 1
        out = torch.empty((N, PD, PH, PW), device=x.device, dtype=x.dtype)
        total = N * PD * PH * PW
        C = self.out_channels
        C_PAD = max(triton.next_power_of_2(C), 2)
        w = self.conv_transpose.weight.contiguous()
        b = self.conv_transpose.bias.contiguous()
        sub = self.subtract.contiguous()
        BLOCK = 128
        grid = (triton.cdiv(total, BLOCK),)
        fused_convt_pool_softmax_swish_max_kernel[grid](
            x, w, b, sub, out,
            total,
            C_in, D, H, W,
            Dc, Hc, Wc,
            PD, PH, PW,
            C_OUT=C, C_PAD=C_PAD,
            K=k, STRIDE=s, PAD=p,
            PK=self.pk, PS=self.ps, PP=self.pp,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
