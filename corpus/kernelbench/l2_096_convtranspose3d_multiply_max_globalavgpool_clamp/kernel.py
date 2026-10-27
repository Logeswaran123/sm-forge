import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _convt_scale_maxpool_partial_kernel(
    x_ptr, w_ptr, b_ptr, part_ptr,
    D_in, H_in, W_in,
    PD, PH, PW,
    n_pos, nblk,
    scale,
    C_IN: tl.constexpr, C_OUT: tl.constexpr, CO_PAD: tl.constexpr,
    K: tl.constexpr, STRIDE: tl.constexpr, PAD: tl.constexpr,
    POOL: tl.constexpr, BLOCK: tl.constexpr,
):
    n = tl.program_id(0)
    blk = tl.program_id(1)

    offs = blk * BLOCK + tl.arange(0, BLOCK)
    pmask = offs < n_pos
    pw = offs % PW
    t = offs // PW
    ph = t % PH
    pd = t // PH

    co = tl.arange(0, CO_PAD)
    comask = co < C_OUT
    bias = tl.load(b_ptr + co, mask=comask, other=0.0).to(tl.float32)

    HW_in = H_in * W_in
    DHW_in = D_in * HW_in
    x_base = x_ptr + n.to(tl.int64) * (C_IN * DHW_in)

    K3: tl.constexpr = K * K * K

    best = tl.full([BLOCK, CO_PAD], float("-inf"), tl.float32)
    for a in tl.static_range(POOL):
        od = pd * POOL + a
        for b in tl.static_range(POOL):
            oh = ph * POOL + b
            for c in tl.static_range(POOL):
                ow = pw * POOL + c
                acc = tl.zeros([BLOCK, CO_PAD], tl.float32) + bias[None, :]
                for kd in tl.static_range(K):
                    td = od + PAD - kd
                    tdc = tl.maximum(td, 0)
                    idd = tdc // STRIDE
                    vd = (td >= 0) & ((tdc % STRIDE) == 0) & (idd < D_in)
                    for kh in tl.static_range(K):
                        th = oh + PAD - kh
                        thc = tl.maximum(th, 0)
                        ih = thc // STRIDE
                        vh = (th >= 0) & ((thc % STRIDE) == 0) & (ih < H_in)
                        for kw in tl.static_range(K):
                            tw = ow + PAD - kw
                            twc = tl.maximum(tw, 0)
                            iw = twc // STRIDE
                            vw = (tw >= 0) & ((twc % STRIDE) == 0) & (iw < W_in)
                            valid = vd & vh & vw & pmask
                            sp_off = idd * HW_in + ih * W_in + iw
                            for ci in tl.static_range(C_IN):
                                xv = tl.load(x_base + ci * DHW_in + sp_off,
                                             mask=valid, other=0.0).to(tl.float32)
                                wv = tl.load(w_ptr + ci * (C_OUT * K3) + co * K3
                                             + (kd * K * K + kh * K + kw),
                                             mask=comask, other=0.0).to(tl.float32)
                                acc += xv[:, None] * wv[None, :]
                best = tl.maximum(best, acc * scale)

    best = tl.where(pmask[:, None], best, 0.0)
    s = tl.sum(best, axis=0)
    tl.store(part_ptr + (n * nblk + blk) * CO_PAD + co, s)


@triton.jit
def _reduce_mean_clamp_kernel(
    part_ptr, out_ptr, nblk, inv_count, cmin, cmax,
    C_OUT: tl.constexpr, CO_PAD: tl.constexpr, RB: tl.constexpr,
):
    n = tl.program_id(0)
    co = tl.arange(0, CO_PAD)
    rb = tl.arange(0, RB)
    total = tl.zeros([CO_PAD], tl.float32)
    for start in range(0, nblk, RB):
        rows = start + rb
        m = rows < nblk
        v = tl.load(part_ptr + (n * nblk + rows[:, None]) * CO_PAD + co[None, :],
                    mask=m[:, None], other=0.0)
        total += tl.sum(v, axis=0)
    mean = total * inv_count
    mean = tl.minimum(tl.maximum(mean, cmin), cmax)
    tl.store(out_ptr + n * C_OUT + co, mean.to(out_ptr.dtype.element_ty), mask=co < C_OUT)


def _triple(v):
    if isinstance(v, (tuple, list)):
        assert len(v) == 3 and v[0] == v[1] == v[2]
        return int(v[0])
    return int(v)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.clamp_min = 0
        self.clamp_max = 1

    def forward(self, x):
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()
        b = self.conv_transpose.bias.contiguous()
        N, C_in, D_in, H_in, W_in = x.shape
        C_out = w.shape[1]
        K = _triple(self.conv_transpose.kernel_size)
        S = _triple(self.conv_transpose.stride)
        P = _triple(self.conv_transpose.padding)
        OP = _triple(self.conv_transpose.output_padding)
        POOL = _triple(self.maxpool.kernel_size)

        D_out = (D_in - 1) * S - 2 * P + K + OP
        H_out = (H_in - 1) * S - 2 * P + K + OP
        W_out = (W_in - 1) * S - 2 * P + K + OP
        PD, PH, PW = D_out // POOL, H_out // POOL, W_out // POOL
        n_pos = PD * PH * PW

        CO_PAD = max(triton.next_power_of_2(C_out), 2)
        BLOCK = 128
        nblk = triton.cdiv(n_pos, BLOCK)

        part = torch.empty((N, nblk, CO_PAD), device=x.device, dtype=torch.float32)
        _convt_scale_maxpool_partial_kernel[(N, nblk)](
            x, w, b, part,
            D_in, H_in, W_in,
            PD, PH, PW,
            n_pos, nblk,
            float(self.scale),
            C_IN=C_in, C_OUT=C_out, CO_PAD=CO_PAD,
            K=K, STRIDE=S, PAD=P, POOL=POOL, BLOCK=BLOCK,
            num_warps=4,
        )

        out = torch.empty((N, C_out, 1, 1, 1), device=x.device, dtype=x.dtype)
        _reduce_mean_clamp_kernel[(N,)](
            part, out, nblk, 1.0 / float(n_pos),
            float(self.clamp_min), float(self.clamp_max),
            C_OUT=C_out, CO_PAD=CO_PAD, RB=32,
            num_warps=4,
        )
        return out
