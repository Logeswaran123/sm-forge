import torch
import torch.nn as nn
import triton
import triton.language as tl


def _triple(v):
    if isinstance(v, (tuple, list)):
        return tuple(int(a) for a in v)
    return (int(v), int(v), int(v))


@triton.jit
def _fused_convt_lse_kernel(
    x_ptr, w_ptr, cb_ptr, b_ptr, out_ptr,
    D, H, W, OD, OH, OW,
    C_IN: tl.constexpr, C_OUT: tl.constexpr, C_PAD: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = tl.program_id(1).to(tl.int64)

    S_out = OD * OH * OW
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < S_out
    ow = offs % OW
    t = offs // OW
    oh = t % OH
    od = t // OH

    co = tl.arange(0, C_PAD)
    cm = co < C_OUT

    DHW = D * H * W
    x_base = x_ptr + n * C_IN * DHW

    KVOL: tl.constexpr = KD * KH * KW

    acc = tl.zeros([BLOCK, C_PAD], dtype=tl.float32)

    for kd in tl.static_range(KD):
        td = od + PD - kd
        idd = td // SD
        vd = (td >= 0) & ((td % SD) == 0) & (idd < D)
        for kh in tl.static_range(KH):
            th = oh + PH - kh
            ih = th // SH
            vh = (th >= 0) & ((th % SH) == 0) & (ih < H)
            for kw in tl.static_range(KW):
                tw = ow + PW - kw
                iw = tw // SW
                vw = (tw >= 0) & ((tw % SW) == 0) & (iw < W)
                valid = m & vd & vh & vw
                sp = idd * (H * W) + ih * W + iw
                sp = tl.where(valid, sp, 0)
                for ci in tl.static_range(C_IN):
                    xv = tl.load(x_base + ci * DHW + sp, mask=valid, other=0.0).to(tl.float32)
                    w_off = ci * C_OUT * KVOL + co * KVOL + (kd * KH * KW + kh * KW + kw)
                    wv = tl.load(w_ptr + w_off, mask=cm, other=0.0).to(tl.float32)
                    acc += xv[:, None] * wv[None, :]

    cb = tl.load(cb_ptr + co, mask=cm, other=0.0).to(tl.float32)
    acc = acc + cb[None, :]
    acc = tl.where(cm[None, :], acc, float("-inf"))

    mx = tl.max(acc, axis=1)
    e = tl.exp(acc - mx[:, None])
    s = tl.sum(e, axis=1)
    lse = mx + tl.log(s)

    sig = 1.0 / (1.0 + tl.exp(-(lse + 3.0)))
    y = lse * sig / 6.0
    b = tl.load(b_ptr).to(tl.float32)
    y = y - b
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)

    tl.store(out_ptr + n * S_out + offs, y.to(out_ptr.dtype.element_ty), mask=m)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = _triple(kernel_size)
        self.s = _triple(stride)
        self.p = _triple(padding)

    def forward(self, x):
        x = x.contiguous()
        N, C_in, D, H, W = x.shape
        KD, KH, KW = self.k
        SD, SH, SW = self.s
        PD, PH, PW = self.p
        OD = (D - 1) * SD - 2 * PD + KD
        OH = (H - 1) * SH - 2 * PH + KH
        OW = (W - 1) * SW - 2 * PW + KW

        w = self.conv_transpose.weight.contiguous()
        cb = self.conv_transpose.bias.contiguous()
        b = self.bias.contiguous()

        out = torch.empty((N, 1, OD, OH, OW), device=x.device, dtype=x.dtype)
        C_out = self.out_channels
        C_PAD = triton.next_power_of_2(C_out)
        BLOCK = 128
        S_out = OD * OH * OW
        grid = (triton.cdiv(S_out, BLOCK), N)
        _fused_convt_lse_kernel[grid](
            x, w, cb, b, out,
            D, H, W, OD, OH, OW,
            C_IN=C_in, C_OUT=C_out, C_PAD=C_PAD,
            KD=KD, KH=KH, KW=KW,
            SD=SD, SH=SH, SW=SW,
            PD=PD, PH=PH, PW=PW,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
