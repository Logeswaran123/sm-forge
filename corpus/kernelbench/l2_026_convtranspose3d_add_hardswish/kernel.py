import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def convt3d_add_hswish_kernel(
    x_ptr, w_ptr, b_ptr, add_ptr, out_ptr,
    N, CIN, COUT,
    D, H, W,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_CO: tl.constexpr, BLOCK_CI: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_co = tl.program_id(1)

    OS = OD * OH * OW
    IS = D * H * W
    total_p = N * OS

    offs_p = (pid_p * BLOCK_P + tl.arange(0, BLOCK_P)).to(tl.int64)
    p_mask = offs_p < total_p
    n = offs_p // OS
    sp = offs_p % OS
    od = sp // (OH * OW)
    rem = sp % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_mask = offs_co < COUT
    offs_ci_base = tl.arange(0, BLOCK_CI)

    acc = tl.zeros((BLOCK_P, BLOCK_CO), dtype=tl.float32)

    for kd in tl.static_range(KD):
        td = od + PD - kd
        td_c = tl.where(td >= 0, td, 0)
        idd = td_c // SD
        vd = (td >= 0) & ((td_c % SD) == 0) & (idd < D)
        for kh in tl.static_range(KH):
            th = oh + PH - kh
            th_c = tl.where(th >= 0, th, 0)
            ih = th_c // SH
            vh = (th >= 0) & ((th_c % SH) == 0) & (ih < H)
            for kw in tl.static_range(KW):
                tw = ow + PW - kw
                tw_c = tl.where(tw >= 0, tw, 0)
                iw = tw_c // SW
                vw = (tw >= 0) & ((tw_c % SW) == 0) & (iw < W)
                valid = p_mask & vd & vh & vw
                sp_in = n * CIN * IS + idd * (H * W) + ih * W + iw
                tap = (kd * KH + kh) * KW + kw
                for ci0 in range(0, CIN, BLOCK_CI):
                    offs_ci = ci0 + offs_ci_base
                    ci_mask = offs_ci < CIN
                    x_ptrs = x_ptr + sp_in[:, None] + offs_ci[None, :].to(tl.int64) * IS
                    xt = tl.load(x_ptrs, mask=valid[:, None] & ci_mask[None, :], other=0.0)
                    w_ptrs = w_ptr + (tap * CIN + offs_ci[:, None]) * COUT + offs_co[None, :]
                    wt = tl.load(w_ptrs, mask=ci_mask[:, None] & co_mask[None, :], other=0.0)
                    acc += tl.dot(xt, wt)

    if HAS_BIAS:
        b = tl.load(b_ptr + offs_co, mask=co_mask, other=0.0).to(tl.float32)
        acc += b[None, :]

    out_off = n[:, None] * COUT * OS + offs_co[None, :].to(tl.int64) * OS + sp[:, None]
    m2 = p_mask[:, None] & co_mask[None, :]
    a = tl.load(add_ptr + out_off, mask=m2, other=0.0).to(tl.float32)
    v = acc + a
    hs = v * tl.minimum(tl.maximum(v + 3.0, 0.0), 6.0) / 6.0
    y = v * hs
    tl.store(out_ptr + out_off, y.to(out_ptr.dtype.element_ty), mask=m2)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                                 padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))  # unused in reference forward

    def forward(self, x, add_input):
        conv = self.conv_transpose
        x = x.contiguous()
        add_input = add_input.contiguous()
        N, CIN, D, H, W = x.shape
        COUT = conv.out_channels
        KD, KH, KW = conv.kernel_size
        SD, SH, SW = conv.stride
        PD, PH, PW = conv.padding
        OPD, OPH, OPW = conv.output_padding
        DLD, DLH, DLW = conv.dilation
        OD = (D - 1) * SD - 2 * PD + DLD * (KD - 1) + OPD + 1
        OH = (H - 1) * SH - 2 * PH + DLH * (KH - 1) + OPH + 1
        OW = (W - 1) * SW - 2 * PW + DLW * (KW - 1) + OPW + 1

        # weight [CIN, COUT, KD, KH, KW] -> [KD, KH, KW, CIN, COUT]
        wp = conv.weight.permute(2, 3, 4, 0, 1).contiguous().to(x.dtype)
        has_bias = conv.bias is not None
        b = conv.bias.contiguous() if has_bias else wp

        out_dtype = torch.promote_types(x.dtype, add_input.dtype)
        out = torch.empty((N, COUT, OD, OH, OW), device=x.device, dtype=out_dtype)

        BLOCK_P = 64
        BLOCK_CO = 64
        BLOCK_CI = max(16, triton.next_power_of_2(min(CIN, 64)))
        grid = (triton.cdiv(N * OD * OH * OW, BLOCK_P), triton.cdiv(COUT, BLOCK_CO))
        convt3d_add_hswish_kernel[grid](
            x, wp, b, add_input, out,
            N, CIN, COUT,
            D, H, W,
            OD, OH, OW,
            KD=KD, KH=KH, KW=KW,
            SD=SD, SH=SH, SW=SW,
            PD=PD, PH=PH, PW=PW,
            HAS_BIAS=has_bias,
            BLOCK_P=BLOCK_P, BLOCK_CO=BLOCK_CO, BLOCK_CI=BLOCK_CI,
            num_warps=4,
        )
        return out
