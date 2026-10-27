import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def convt3d_kernel(
    x_ptr, w_ptr, y_ptr, sum_ptr,
    CIN, COUT, D, H, W, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_CO: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_c = tl.program_id(1)
    n = tl.program_id(2)

    S = OD * OH * OW
    offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    smask = offs < S
    ow = offs % OW
    t = offs // OW
    oh = t % OH
    od = t // OH

    co = pid_c * BLOCK_CO + tl.arange(0, BLOCK_CO)
    cmask = co < COUT

    K3: tl.constexpr = KD * KH * KW
    DHW = D * H * W
    x_base = x_ptr + n.to(tl.int64) * CIN * DHW

    acc = tl.zeros((BLOCK_S, BLOCK_CO), dtype=tl.float32)

    for kd in tl.static_range(KD):
        td = od + PD - kd
        tdc = tl.where(td >= 0, td, 0)
        idd = tdc // SD
        vd = (td >= 0) & ((tdc % SD) == 0) & (idd < D)
        for kh in tl.static_range(KH):
            th = oh + PH - kh
            thc = tl.where(th >= 0, th, 0)
            ih = thc // SH
            vh = (th >= 0) & ((thc % SH) == 0) & (ih < H)
            for kw in tl.static_range(KW):
                tw = ow + PW - kw
                twc = tl.where(tw >= 0, tw, 0)
                iw = twc // SW
                vw = (tw >= 0) & ((twc % SW) == 0) & (iw < W)
                valid = smask & vd & vh & vw
                in_off = tl.where(valid, (idd * H + ih) * W + iw, 0)
                tap = (kd * KH + kh) * KW + kw
                for ci in range(0, CIN):
                    xv = tl.load(x_base + ci * DHW + in_off, mask=valid, other=0.0).to(tl.float32)
                    wv = tl.load(w_ptr + (ci * COUT + co) * K3 + tap, mask=cmask, other=0.0).to(tl.float32)
                    acc += xv[:, None] * wv[None, :]

    full_mask = smask[:, None] & cmask[None, :]
    y_ptrs = y_ptr + (n.to(tl.int64) * COUT + co[None, :]) * S + offs[:, None]
    tl.store(y_ptrs, acc, mask=full_mask)

    part = tl.sum(tl.where(full_mask, acc, 0.0), axis=0)
    tl.atomic_add(sum_ptr + n * COUT + co, part, mask=cmask)


@triton.jit
def bn_sub_mean_kernel(
    y_ptr, sum_ptr, gamma_ptr, var_ptr, out_ptr,
    COUT, S, eps,
    BLOCK: tl.constexpr,
):
    pid_s = tl.program_id(0)
    nc = tl.program_id(1)
    c = nc % COUT

    total = tl.load(sum_ptr + nc)
    mean = total / S
    g = tl.load(gamma_ptr + c).to(tl.float32)
    v = tl.load(var_ptr + c).to(tl.float32)
    scale = g / tl.sqrt(v + eps)

    offs = pid_s * BLOCK + tl.arange(0, BLOCK)
    m = offs < S
    base = nc.to(tl.int64) * S
    y = tl.load(y_ptr + base + offs, mask=m, other=0.0)
    o = (y - mean) * scale
    tl.store(out_ptr + base + offs, o.to(out_ptr.dtype.element_ty), mask=m)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        conv = self.conv_transpose
        bn = self.batch_norm
        x = x.contiguous()
        w = conv.weight.contiguous()
        N, CIN, D, H, W = x.shape
        COUT = w.shape[1]
        KD, KH, KW = conv.kernel_size
        SD, SH, SW = conv.stride
        PD, PH, PW = conv.padding
        OPD, OPH, OPW = conv.output_padding
        OD = (D - 1) * SD - 2 * PD + KD + OPD
        OH = (H - 1) * SH - 2 * PH + KH + OPH
        OW = (W - 1) * SW - 2 * PW + KW + OPW
        S = OD * OH * OW

        y = torch.empty((N, COUT, OD, OH, OW), device=x.device, dtype=torch.float32)
        sums = torch.zeros((N * COUT,), device=x.device, dtype=torch.float32)

        BLOCK_S = 128
        BLOCK_CO = min(32, triton.next_power_of_2(COUT))
        grid = (triton.cdiv(S, BLOCK_S), triton.cdiv(COUT, BLOCK_CO), N)
        convt3d_kernel[grid](
            x, w, y, sums,
            CIN, COUT, D, H, W, OD, OH, OW,
            KD=KD, KH=KH, KW=KW, SD=SD, SH=SH, SW=SW, PD=PD, PH=PH, PW=PW,
            BLOCK_S=BLOCK_S, BLOCK_CO=BLOCK_CO,
            num_warps=4,
        )

        out = torch.empty((N, COUT, OD, OH, OW), device=x.device, dtype=x.dtype)
        BLOCK = 1024
        grid2 = (triton.cdiv(S, BLOCK), N * COUT)
        bn_sub_mean_kernel[grid2](
            y, sums, bn.weight, bn.running_var, out,
            COUT, S, float(bn.eps),
            BLOCK=BLOCK, num_warps=4,
        )
        return out
