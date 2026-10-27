import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _convT_pool_kernel(
    x_ptr, w_ptr, cb_ptr, s1_ptr, b_ptr, s2_ptr, out_ptr,
    CIN, COUT, D, H, W, PD, PH, PW,
    K: tl.constexpr, S: tl.constexpr, P: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nc = tl.program_id(1)
    n = nc // COUT
    co = nc % COUT

    total = PD * PH * PW
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < total
    pw = offs % PW
    ph = (offs // PW) % PH
    pd = offs // (PW * PH)

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    DHW = D * H * W

    for ci in range(CIN):
        x_base = x_ptr + (n * CIN + ci) * DHW
        for kd in tl.static_range(K):
            for kh in tl.static_range(K):
                for kw in tl.static_range(K):
                    w_idx = (((ci * COUT + co) * K + kd) * K + kh) * K + kw
                    wv = tl.load(w_ptr + w_idx).to(tl.float32)
                    for sd in tl.static_range(2):
                        td = pd * 2 + sd + P - kd
                        idd = td // S
                        vd = (td >= 0) & ((td % S) == 0) & (idd < D)
                        idd = tl.where(vd, idd, 0)
                        for sh in tl.static_range(2):
                            th = ph * 2 + sh + P - kh
                            ih = th // S
                            vh = (th >= 0) & ((th % S) == 0) & (ih < H)
                            ih = tl.where(vh, ih, 0)
                            for sw in tl.static_range(2):
                                tw = pw * 2 + sw + P - kw
                                iw = tw // S
                                vw = (tw >= 0) & ((tw % S) == 0) & (iw < W)
                                iw = tl.where(vw, iw, 0)
                                valid = m & vd & vh & vw
                                xv = tl.load(x_base + (idd * H + ih) * W + iw,
                                             mask=valid, other=0.0).to(tl.float32)
                                acc += xv * wv

    cb = tl.load(cb_ptr + co).to(tl.float32)
    s1 = tl.load(s1_ptr).to(tl.float32)
    bb = tl.load(b_ptr + co).to(tl.float32)
    s2 = tl.load(s2_ptr).to(tl.float32)
    avg = acc * 0.125 + cb
    res = (avg * s1 + bb) * s2
    tl.store(out_ptr + nc * total + offs, res.to(out_ptr.dtype.element_ty), mask=m)


def _first(v):
    if isinstance(v, (tuple, list)):
        return v[0]
    return v


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))
        self.k = _first(kernel_size)
        self.s = _first(stride)
        self.p = _first(padding)

    def forward(self, x):
        x = x.contiguous()
        N, CIN, D, H, W = x.shape
        w = self.conv_transpose.weight.contiguous()
        cb = self.conv_transpose.bias.contiguous()
        COUT = w.shape[1]
        K, S, P = self.k, self.s, self.p
        OD = (D - 1) * S - 2 * P + K
        OH = (H - 1) * S - 2 * P + K
        OW = (W - 1) * S - 2 * P + K
        PD, PH, PW = OD // 2, OH // 2, OW // 2
        out = torch.empty((N, COUT, PD, PH, PW), device=x.device, dtype=x.dtype)
        total = PD * PH * PW
        if total == 0 or N == 0:
            return out
        BLOCK = 256
        grid = (triton.cdiv(total, BLOCK), N * COUT)
        _convT_pool_kernel[grid](
            x, w, cb, self.scale1, self.bias.contiguous(), self.scale2, out,
            CIN, COUT, D, H, W, PD, PH, PW,
            K=K, S=S, P=P, BLOCK=BLOCK,
        )
        return out
