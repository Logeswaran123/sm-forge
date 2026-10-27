import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def convt3d_relu_kernel(
    x_ptr, w_ptr, y_ptr,
    CIN, COUT, D, H, W, OD, OH, OW,
    KS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    n = tl.program_id(1)
    pid_n = tl.program_id(2)

    ODHW = OD * OH * OW
    DHW = D * H * W
    HW = H * W

    offs_m = pid_m * BM + tl.arange(0, BM)
    m_mask = offs_m < ODHW
    ow = offs_m % OW
    t = offs_m // OW
    oh = t % OH
    od = t // OH

    offs_n = pid_n * BN + tl.arange(0, BN)
    n_mask = offs_n < COUT
    offs_k = tl.arange(0, BK)

    x_base = x_ptr + n.to(tl.int64) * CIN * DHW

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for tap in range(KS * KS * KS):
        kd = tap // (KS * KS)
        kh = (tap // KS) % KS
        kw = tap % KS
        idd = od - kd
        ih = oh - kh
        iw = ow - kw
        valid = m_mask & (idd >= 0) & (idd < D) & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        sp = idd * HW + ih * W + iw
        for c0 in range(0, CIN, BK):
            ci = c0 + offs_k
            k_mask = ci < CIN
            a = tl.load(
                x_base + ci[None, :] * DHW + sp[:, None],
                mask=valid[:, None] & k_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                w_ptr + (tap * CIN + ci)[:, None] * COUT + offs_n[None, :],
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc = tl.dot(a.to(tl.float16), b.to(tl.float16), acc)

    acc = tl.maximum(acc, 0.0)
    y_base = y_ptr + n.to(tl.int64) * COUT * ODHW
    y_ptrs = y_base + offs_n[None, :] * ODHW + offs_m[:, None]
    tl.store(y_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def groupnorm_kernel(
    y_ptr, out_ptr, gamma_ptr, beta_ptr,
    C, G, S, eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G
    CPG = C // G
    L = CPG * S
    base = (n * C + g * CPG).to(tl.int64) * S
    offs = tl.arange(0, BLOCK)

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, L, BLOCK):
        idx = start + offs
        msk = idx < L
        v = tl.load(y_ptr + base + idx, mask=msk, other=0.0)
        acc += v
    mean = tl.sum(acc, axis=0) / L

    acc2 = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, L, BLOCK):
        idx = start + offs
        msk = idx < L
        v = tl.load(y_ptr + base + idx, mask=msk, other=0.0)
        d = tl.where(msk, v - mean, 0.0)
        acc2 += d * d
    var = tl.sum(acc2, axis=0) / L
    rstd = 1.0 / tl.sqrt(var + eps)

    for start in range(0, L, BLOCK):
        idx = start + offs
        msk = idx < L
        v = tl.load(y_ptr + base + idx, mask=msk, other=0.0)
        c = g * CPG + idx // S
        gam = tl.load(gamma_ptr + c, mask=msk, other=1.0).to(tl.float32)
        bet = tl.load(beta_ptr + c, mask=msk, other=0.0).to(tl.float32)
        o = (v - mean) * rstd * gam + bet
        tl.store(out_ptr + base + idx, o.to(out_ptr.dtype.element_ty), mask=msk)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.relu = nn.ReLU()
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)

    def forward(self, x):
        x = x.contiguous()
        N, CIN, D, H, W = x.shape
        w = self.conv_transpose.weight  # [CIN, COUT, k, k, k]
        COUT = w.shape[1]
        KS = w.shape[2]
        OD, OH, OW = D + KS - 1, H + KS - 1, W + KS - 1
        ODHW = OD * OH * OW

        # [k, k, k, CIN, COUT] contiguous
        wp = w.permute(2, 3, 4, 0, 1).contiguous()

        y = torch.empty((N, COUT, OD, OH, OW), device=x.device, dtype=torch.float32)
        BM, BN, BK = 128, 128, 32
        grid = (triton.cdiv(ODHW, BM), N, triton.cdiv(COUT, BN))
        convt3d_relu_kernel[grid](
            x, wp, y,
            CIN, COUT, D, H, W, OD, OH, OW,
            KS=KS, BM=BM, BN=BN, BK=BK,
            num_warps=8, num_stages=2,
        )

        G = self.group_norm.num_groups
        out = torch.empty((N, COUT, OD, OH, OW), device=x.device, dtype=x.dtype)
        groupnorm_kernel[(N * G,)](
            y, out, self.group_norm.weight, self.group_norm.bias,
            COUT, G, ODHW, float(self.group_norm.eps),
            BLOCK=4096, num_warps=8,
        )
        return out
