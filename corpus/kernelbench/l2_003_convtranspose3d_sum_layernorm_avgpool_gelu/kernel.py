import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def convt_ln_kernel(
    x_ptr, w_ptr, b_ptr, sw_ptr, g_ptr, beta_ptr, y_ptr,
    CI, CO, DIN, HIN, WIN, DOUT, HOUT, WOUT,
    eps,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    DD: tl.constexpr, DH: tl.constexpr, DW: tl.constexpr,
    BLOCK_CO: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_co = tl.program_id(1)

    oh = pid_row % HOUT
    tmp = pid_row // HOUT
    od = tmp % DOUT
    n = tmp // DOUT

    co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_mask = co < CO
    ow = tl.arange(0, BLOCK_W)
    ow_mask = ow < WOUT

    acc = tl.zeros((BLOCK_CO, BLOCK_W), dtype=tl.float32)

    HW_IN = HIN * WIN
    for ci in range(0, CI):
        x_base = x_ptr + (n * CI + ci).to(tl.int64) * DIN * HW_IN
        w_base = w_ptr + (ci * CO + co) * (KD * KH * KW)
        for kd in tl.static_range(KD):
            nd = od + PD - kd * DD
            idd = nd // SD
            d_ok = (nd >= 0) & ((nd % SD) == 0) & (idd < DIN)
            for kh in tl.static_range(KH):
                nh = oh + PH - kh * DH
                ih = nh // SH
                h_ok = (nh >= 0) & ((nh % SH) == 0) & (ih < HIN)
                dh_ok = d_ok & h_ok
                for kw in tl.static_range(KW):
                    nw = ow + PW - kw * DW
                    iw = nw // SW
                    w_ok = (nw >= 0) & ((nw % SW) == 0) & (iw < WIN) & ow_mask
                    m = w_ok & dh_ok
                    xoff = idd * HW_IN + ih * WIN + iw
                    xoff = tl.where(m, xoff, 0)
                    xv = tl.load(x_base + xoff, mask=m, other=0.0).to(tl.float32)
                    wv = tl.load(w_base + (kd * KH + kh) * KW + kw,
                                 mask=co_mask, other=0.0).to(tl.float32)
                    acc += wv[:, None] * xv[None, :]

    bias = tl.load(b_ptr + co, mask=co_mask, other=0.0).to(tl.float32)
    sw = tl.load(sw_ptr).to(tl.float32)
    acc = acc + bias[:, None] + sw

    # LayerNorm over the last (W) dimension
    wm2 = ow_mask[None, :]
    accm = tl.where(wm2, acc, 0.0)
    mean = tl.sum(accm, axis=1) / WOUT
    diff = tl.where(wm2, acc - mean[:, None], 0.0)
    var = tl.sum(diff * diff, axis=1) / WOUT
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(g_ptr + ow, mask=ow_mask, other=0.0).to(tl.float32)
    bt = tl.load(beta_ptr + ow, mask=ow_mask, other=0.0).to(tl.float32)
    y = diff * rstd[:, None] * g[None, :] + bt[None, :]

    row_off = ((n * CO + co).to(tl.int64) * DOUT + od) * HOUT + oh
    yptrs = y_ptr + row_off[:, None] * WOUT + ow[None, :]
    tl.store(yptrs, y, mask=co_mask[:, None] & wm2)


@triton.jit
def pool_gelu_kernel(
    y_ptr, out_ptr, TOTAL,
    D, H, W, PDo, PHo, PWo,
    KPD: tl.constexpr, KPH: tl.constexpr, KPW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    pw = offs % PWo
    t = offs // PWo
    ph = t % PHo
    t = t // PHo
    pd = t % PDo
    nc = t // PDo

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for kd in tl.static_range(KPD):
        for kh in tl.static_range(KPH):
            for kw in tl.static_range(KPW):
                d = pd * KPD + kd
                h = ph * KPH + kh
                w = pw * KPW + kw
                off = ((nc * D + d) * H + h) * W + w
                acc += tl.load(y_ptr + off, mask=mask, other=0.0)
    acc = acc / (KPD * KPH * KPW)
    g = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    tl.store(out_ptr + offs, g.to(out_ptr.dtype.element_ty), mask=mask)


def _triple(v):
    if isinstance(v, int):
        return (v, v, v)
    return tuple(v)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding,
                                                 output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        conv = self.conv_transpose
        x = x.contiguous()
        N, CI, DIN, HIN, WIN = x.shape
        CO = conv.out_channels
        KD, KH, KW = _triple(conv.kernel_size)
        SD, SH, SW = _triple(conv.stride)
        PD, PH, PW = _triple(conv.padding)
        OPD, OPH, OPW = _triple(conv.output_padding)
        DD, DH, DW = _triple(conv.dilation)

        DOUT = (DIN - 1) * SD - 2 * PD + DD * (KD - 1) + OPD + 1
        HOUT = (HIN - 1) * SH - 2 * PH + DH * (KH - 1) + OPH + 1
        WOUT = (WIN - 1) * SW - 2 * PW + DW * (KW - 1) + OPW + 1

        weight = conv.weight.contiguous()
        bias = conv.bias.contiguous()
        gamma = self.norm.weight.contiguous()
        beta = self.norm.bias.contiguous()

        y = torch.empty((N, CO, DOUT, HOUT, WOUT), device=x.device, dtype=torch.float32)
        BLOCK_W = triton.next_power_of_2(WOUT)
        BLOCK_CO = 32 if BLOCK_W <= 64 else 16
        grid = (N * DOUT * HOUT, triton.cdiv(CO, BLOCK_CO))
        convt_ln_kernel[grid](
            x, weight, bias, self.sum_weight, gamma, beta, y,
            CI, CO, DIN, HIN, WIN, DOUT, HOUT, WOUT,
            float(self.norm.eps),
            KD=KD, KH=KH, KW=KW, SD=SD, SH=SH, SW=SW,
            PD=PD, PH=PH, PW=PW, DD=DD, DH=DH, DW=DW,
            BLOCK_CO=BLOCK_CO, BLOCK_W=BLOCK_W,
            num_warps=4,
        )

        KPD, KPH, KPW = _triple(self.avg_pool.kernel_size)
        PDo, PHo, PWo = DOUT // KPD, HOUT // KPH, WOUT // KPW
        out = torch.empty((N, CO, PDo, PHo, PWo), device=x.device, dtype=x.dtype)
        total = out.numel()
        BLOCK = 256
        pool_gelu_kernel[(triton.cdiv(total, BLOCK),)](
            y, out, total, DOUT, HOUT, WOUT, PDo, PHo, PWo,
            KPD=KPD, KPH=KPH, KPW=KPW, BLOCK=BLOCK, num_warps=4,
        )
        return out
