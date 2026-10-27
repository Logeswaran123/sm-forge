import torch
import torch.nn as nn
import triton
import triton.language as tl


def _triple(v):
    if isinstance(v, (tuple, list)):
        return tuple(int(a) for a in v)
    return (int(v), int(v), int(v))


@triton.jit
def _convt3d_clamp_div_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, CIN, COUT,
    ID, IH, IW,
    OD, OH, OW,
    min_value, divisor,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    S_out = OD * OH * OW
    S_in = ID * IH * IW
    M = N * S_out

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < COUT

    n_idx = offs_m // S_out
    sp = offs_m % S_out
    od = sp // (OH * OW)
    rem = sp % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    x_batch_base = n_idx.to(tl.int64) * CIN * S_in

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    KVOL: tl.constexpr = KD * KH * KW

    for kd in tl.static_range(KD):
        td = od + PD - kd
        vd = (td >= 0) & ((td % SD) == 0)
        idd = td // SD
        vd = vd & (idd < ID)
        for kh in tl.static_range(KH):
            th = oh + PH - kh
            vh = (th >= 0) & ((th % SH) == 0)
            ihh = th // SH
            vh = vh & (ihh < IH)
            for kw in tl.static_range(KW):
                tw = ow + PW - kw
                vw = (tw >= 0) & ((tw % SW) == 0)
                iww = tw // SW
                vw = vw & (iww < IW)
                valid = m_mask & vd & vh & vw
                sp_in = idd * (IH * IW) + ihh * IW + iww
                sp_in = tl.where(valid, sp_in, 0)
                row_off = x_batch_base + sp_in.to(tl.int64)
                tap = kd * (KH * KW) + kh * KW + kw
                for k0 in range(0, CIN, BLOCK_K):
                    offs_k = k0 + tl.arange(0, BLOCK_K)
                    k_mask = offs_k < CIN
                    a_ptrs = x_ptr + row_off[:, None] + offs_k[None, :].to(tl.int64) * S_in
                    a = tl.load(a_ptrs, mask=valid[:, None] & k_mask[None, :], other=0.0)
                    # weight layout [CIN, COUT, KD, KH, KW]
                    b_ptrs = w_ptr + offs_k[:, None] * (COUT * KVOL) + offs_n[None, :] * KVOL + tap
                    b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
                    b = b.to(a.dtype)
                    acc = tl.dot(a, b, acc)

    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]
    acc = tl.maximum(acc, min_value)
    acc = acc / divisor

    out_off = (n_idx.to(tl.int64) * COUT * S_out + sp.to(tl.int64))[:, None] \
        + offs_n[None, :].to(tl.int64) * S_out
    tl.store(out_ptr + out_off, acc.to(out_ptr.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

    def forward(self, x):
        conv = self.conv_transpose
        x = x.contiguous()
        w = conv.weight.contiguous()
        b = conv.bias
        N, CIN, ID, IH, IW = x.shape
        COUT = w.shape[1]
        KD, KH, KW = w.shape[2], w.shape[3], w.shape[4]
        SD, SH, SW = _triple(conv.stride)
        PD, PH, PW = _triple(conv.padding)
        OPD, OPH, OPW = _triple(conv.output_padding)
        OD = (ID - 1) * SD - 2 * PD + KD + OPD
        OH = (IH - 1) * SH - 2 * PH + KH + OPH
        OW = (IW - 1) * SW - 2 * PW + KW + OPW

        out = torch.empty((N, COUT, OD, OH, OW), device=x.device, dtype=x.dtype)
        has_bias = b is not None
        b_arg = b.contiguous() if has_bias else w

        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        M = N * OD * OH * OW
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(COUT, BLOCK_N))
        _convt3d_clamp_div_kernel[grid](
            x, w, b_arg, out,
            N, CIN, COUT,
            ID, IH, IW,
            OD, OH, OW,
            float(self.min_value), float(self.divisor),
            SD=SD, SH=SH, SW=SW,
            PD=PD, PH=PH, PW=PW,
            KD=KD, KH=KH, KW=KW,
            HAS_BIAS=has_bias,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )
        return out
