import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_pool_partial_kernel(
    x_ptr, w_ptr, cb_ptr, part_ptr,
    Cin, D, H, W,
    Cout, PDo, PHo, PWo, NBLK,
    inv_div,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_P: tl.constexpr, COUT_P: tl.constexpr,
):
    b = tl.program_id(0)
    pid = tl.program_id(1)

    P_total = PDo * PHo * PWo
    offs_p = pid * BLOCK_P + tl.arange(0, BLOCK_P)
    mask_p = offs_p < P_total
    pw = offs_p % PWo
    tmp = offs_p // PWo
    ph = tmp % PHo
    pd = tmp // PHo

    offs_c = tl.arange(0, COUT_P)
    mask_c = offs_c < Cout

    cbias = tl.load(cb_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)

    HW = H * W
    DHW = D * HW
    x_base = x_ptr + b.to(tl.int64) * Cin * DHW

    KVOL: tl.constexpr = KD * KH * KW
    NSUB: tl.constexpr = PD * PH * PW

    mx = tl.full([BLOCK_P, COUT_P], float("-inf"), tl.float32)
    for sub in range(NSUB):
        sd = sub // (PH * PW)
        sh = (sub // PW) % PH
        sw = sub % PW
        od = pd * PD + sd
        oh = ph * PH + sh
        ow = pw * PW + sw
        conv = tl.zeros([BLOCK_P, COUT_P], tl.float32)
        for ci in range(Cin):
            x_ci = x_base + ci * DHW
            w_ci = w_ptr + offs_c * (Cin * KVOL) + ci * KVOL
            for kd in tl.static_range(KD):
                for kh in tl.static_range(KH):
                    for kw in tl.static_range(KW):
                        xoff = (od + kd) * HW + (oh + kh) * W + (ow + kw)
                        xv = tl.load(x_ci + xoff, mask=mask_p, other=0.0).to(tl.float32)
                        wv = tl.load(w_ci + (kd * KH * KW + kh * KW + kw),
                                     mask=mask_c, other=0.0).to(tl.float32)
                        conv += xv[:, None] * wv[None, :]
        conv = (conv + cbias[None, :]) * inv_div
        mx = tl.maximum(mx, conv)

    valid = mask_p[:, None] & mask_c[None, :]
    mx = tl.where(valid, mx, 0.0)
    s = tl.sum(tl.sum(mx, axis=1), axis=0)
    tl.store(part_ptr + b * NBLK + pid, s)


@triton.jit
def finalize_kernel(
    part_ptr, bias_ptr, out_ptr,
    NBLK, Cout, inv_count,
    BLOCK_N: tl.constexpr, COUT_P: tl.constexpr,
):
    b = tl.program_id(0)
    acc = tl.zeros([BLOCK_N], tl.float32)
    for start in range(0, NBLK, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        m = offs < NBLK
        acc += tl.load(part_ptr + b * NBLK + offs, mask=m, other=0.0)
    total = tl.sum(acc, axis=0) * inv_count
    offs_c = tl.arange(0, COUT_P)
    bv = tl.load(bias_ptr + offs_c, mask=offs_c < Cout, other=0.0).to(tl.float32)
    total = total + tl.sum(bv, axis=0)
    tl.store(out_ptr + b, total.to(out_ptr.dtype.element_ty))


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

    def forward(self, x):
        x = x.contiguous()
        B, Cin, D, H, W = x.shape
        w = self.conv.weight.contiguous()
        cb = self.conv.bias.contiguous()
        Cout = w.shape[0]
        KD, KH, KW = w.shape[2], w.shape[3], w.shape[4]

        ps = self.max_pool.kernel_size
        if isinstance(ps, int):
            ps = (ps, ps, ps)
        PD, PH, PW = ps

        Do, Ho, Wo = D - KD + 1, H - KH + 1, W - KW + 1
        PDo, PHo, PWo = Do // PD, Ho // PH, Wo // PW
        P_total = PDo * PHo * PWo

        BLOCK_P = 128
        COUT_P = triton.next_power_of_2(Cout)
        NBLK = triton.cdiv(P_total, BLOCK_P)

        partials = torch.empty((B, NBLK), device=x.device, dtype=torch.float32)
        conv_pool_partial_kernel[(B, NBLK)](
            x, w, cb, partials,
            Cin, D, H, W,
            Cout, PDo, PHo, PWo, NBLK,
            1.0 / float(self.divisor),
            KD=KD, KH=KH, KW=KW, PD=PD, PH=PH, PW=PW,
            BLOCK_P=BLOCK_P, COUT_P=COUT_P,
            num_warps=4,
        )

        out = torch.empty((B, 1, 1, 1), device=x.device, dtype=x.dtype)
        bias_flat = self.bias.reshape(-1).contiguous()
        finalize_kernel[(B,)](
            partials, bias_flat, out,
            NBLK, Cout, 1.0 / float(P_total),
            BLOCK_N=128, COUT_P=COUT_P,
            num_warps=4,
        )
        return out
