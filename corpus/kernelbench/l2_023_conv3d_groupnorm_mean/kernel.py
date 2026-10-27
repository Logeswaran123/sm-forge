import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_stats_kernel(
    x_ptr, w_ptr, b_ptr, part_ptr,
    D, H, W, OD, OH, OW, L, NB, C,
    CIN: tl.constexpr, K: tl.constexpr, C_PAD: tl.constexpr, BLOCK_S: tl.constexpr,
):
    blk = tl.program_id(0)
    n = tl.program_id(1)

    offs_s = blk * BLOCK_S + tl.arange(0, BLOCK_S)
    smask = offs_s < L
    offs_c = tl.arange(0, C_PAD)
    cmask = offs_c < C

    ohw = OH * OW
    od = offs_s // ohw
    rem = offs_s % ohw
    oh = rem // OW
    ow = rem % OW

    HW = H * W
    DHW = D * HW
    x_base = x_ptr + n.to(tl.int64) * (CIN * DHW)
    sp_base = od * HW + oh * W + ow

    K3: tl.constexpr = K * K * K
    acc = tl.zeros([BLOCK_S, C_PAD], dtype=tl.float32)
    for ci in tl.static_range(CIN):
        for kd in tl.static_range(K):
            for kh in tl.static_range(K):
                for kw in tl.static_range(K):
                    xv = tl.load(x_base + ci * DHW + sp_base + kd * HW + kh * W + kw,
                                 mask=smask, other=0.0).to(tl.float32)
                    wv = tl.load(w_ptr + offs_c * (CIN * K3) + ci * K3 + kd * K * K + kh * K + kw,
                                 mask=cmask, other=0.0).to(tl.float32)
                    acc += xv[:, None] * wv[None, :]

    bias = tl.load(b_ptr + offs_c, mask=cmask, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    acc = tl.where(smask[:, None] & cmask[None, :], acc, 0.0)

    s1 = tl.sum(acc, axis=0)
    s2 = tl.sum(acc * acc, axis=0)
    base = part_ptr + (n * NB + blk) * (2 * C_PAD)
    tl.store(base + offs_c, s1)
    tl.store(base + C_PAD + offs_c, s2)


@triton.jit
def gn_mean_kernel(
    part_ptr, g_ptr, beta_ptr, out_ptr,
    NB, L, C, CPG, eps,
    C_PAD: tl.constexpr, BLOCK_B: tl.constexpr,
):
    n = tl.program_id(0)
    offs_c = tl.arange(0, C_PAD)
    cmask = offs_c < C
    offs_b = tl.arange(0, BLOCK_B)

    S = tl.zeros([C_PAD], dtype=tl.float32)
    Q = tl.zeros([C_PAD], dtype=tl.float32)
    for b0 in range(0, NB, BLOCK_B):
        bb = b0 + offs_b
        bmask = bb < NB
        ptrs = part_ptr + (n * NB + bb)[:, None] * (2 * C_PAD) + offs_c[None, :]
        m2 = bmask[:, None]
        p1 = tl.load(ptrs, mask=m2, other=0.0)
        p2 = tl.load(ptrs + C_PAD, mask=m2, other=0.0)
        S += tl.sum(p1, axis=0)
        Q += tl.sum(p2, axis=0)

    gid = offs_c // CPG
    gmask = (gid[:, None] == gid[None, :]) & cmask[:, None] & cmask[None, :]
    gS = tl.sum(tl.where(gmask, S[None, :], 0.0), axis=1)
    gQ = tl.sum(tl.where(gmask, Q[None, :], 0.0), axis=1)

    Lf = L.to(tl.float32)
    cnt = Lf * CPG
    mu = gS / cnt
    var = gQ / cnt - mu * mu
    var = tl.maximum(var, 0.0)
    rstd = 1.0 / tl.sqrt(var + eps)
    meanc = S / Lf

    gamma = tl.load(g_ptr + offs_c, mask=cmask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_c, mask=cmask, other=0.0).to(tl.float32)
    val = gamma * (meanc - mu) * rstd + beta
    val = tl.where(cmask, val, 0.0)
    res = tl.sum(val, axis=0) / C
    tl.store(out_ptr + n, res.to(out_ptr.dtype.element_ty))


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = x.contiguous()
        N, CIN, D, H, W = x.shape
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        C = w.shape[0]
        K = w.shape[2]
        OD, OH, OW = D - K + 1, H - K + 1, W - K + 1
        L = OD * OH * OW
        C_PAD = max(triton.next_power_of_2(C), 2)
        BLOCK_S = 256
        NB = triton.cdiv(L, BLOCK_S)

        part = torch.empty((N, NB, 2, C_PAD), device=x.device, dtype=torch.float32)
        conv3d_stats_kernel[(NB, N)](
            x, w, b, part,
            D, H, W, OD, OH, OW, L, NB, C,
            CIN=CIN, K=K, C_PAD=C_PAD, BLOCK_S=BLOCK_S,
            num_warps=4,
        )

        G = self.group_norm.num_groups
        CPG = C // G
        out = torch.empty((N,), device=x.device, dtype=x.dtype)
        gn_mean_kernel[(N,)](
            part, self.group_norm.weight, self.group_norm.bias, out,
            NB, L, C, CPG, float(self.group_norm.eps),
            C_PAD=C_PAD, BLOCK_B=64,
            num_warps=4,
        )
        return out
