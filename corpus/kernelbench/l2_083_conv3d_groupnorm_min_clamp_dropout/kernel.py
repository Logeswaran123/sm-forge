import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, psum_ptr, psq_ptr,
    D, H, W, OD, OH, OW, S, NB,
    CIN: tl.constexpr, COUT: tl.constexpr, COUT_P: tl.constexpr,
    K: tl.constexpr, BLOCK: tl.constexpr,
):
    n = tl.program_id(0)
    pid = tl.program_id(1)
    s = pid * BLOCK + tl.arange(0, BLOCK)
    smask = s < S
    ohw = OH * OW
    od = s // ohw
    rem = s % ohw
    oh = rem // OW
    ow = rem % OW
    HW = H * W
    DHW = D * HW
    base = n.to(tl.int64) * CIN * DHW + od * HW + oh * W + ow

    co = tl.arange(0, COUT_P)
    cmask = co < COUT
    K3: tl.constexpr = K * K * K

    acc = tl.zeros((BLOCK, COUT_P), dtype=tl.float32)
    for ci in tl.static_range(CIN):
        for kd in tl.static_range(K):
            for kh in tl.static_range(K):
                for kw in tl.static_range(K):
                    xv = tl.load(x_ptr + base + ci * DHW + kd * HW + kh * W + kw,
                                 mask=smask, other=0.0).to(tl.float32)
                    wv = tl.load(w_ptr + co * (CIN * K3) + ci * K3 + kd * K * K + kh * K + kw,
                                 mask=cmask, other=0.0).to(tl.float32)
                    acc += xv[:, None] * wv[None, :]
    bv = tl.load(b_ptr + co, mask=cmask, other=0.0).to(tl.float32)
    acc += bv[None, :]

    m2 = smask[:, None] & cmask[None, :]
    y_off = n.to(tl.int64) * COUT * S + co[None, :] * S + s[:, None]
    tl.store(y_ptr + y_off, acc, mask=m2)

    accm = tl.where(m2, acc, 0.0)
    ps = tl.sum(accm, axis=0)
    pq = tl.sum(accm * accm, axis=0)
    p_off = (n * NB + pid) * COUT + co
    tl.store(psum_ptr + p_off, ps, mask=cmask)
    tl.store(psq_ptr + p_off, pq, mask=cmask)


@triton.jit
def gn_stats_kernel(
    psum_ptr, psq_ptr, mean_ptr, rstd_ptr,
    NB, S, eps,
    COUT: tl.constexpr, G: tl.constexpr, CPG: tl.constexpr, CPG_P: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G
    rb = tl.arange(0, BLOCK_R)
    j = tl.arange(0, CPG_P)
    jmask = j < CPG
    acc_s = tl.zeros((BLOCK_R, CPG_P), dtype=tl.float32)
    acc_q = tl.zeros((BLOCK_R, CPG_P), dtype=tl.float32)
    for start in range(0, NB, BLOCK_R):
        b = start + rb
        m = (b < NB)[:, None] & jmask[None, :]
        off = (n * NB + b)[:, None] * COUT + g * CPG + j[None, :]
        acc_s += tl.load(psum_ptr + off, mask=m, other=0.0)
        acc_q += tl.load(psq_ptr + off, mask=m, other=0.0)
    tot_s = tl.sum(tl.sum(acc_s, axis=1), axis=0)
    tot_q = tl.sum(tl.sum(acc_q, axis=1), axis=0)
    cnt = (S * CPG).to(tl.float32)
    mean = tot_s / cnt
    var = tot_q / cnt - mean * mean
    var = tl.maximum(var, 0.0)
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)


@triton.jit
def gn_apply_kernel(
    y_ptr, out_ptr, mean_ptr, rstd_ptr, gamma_ptr, beta_ptr,
    S, min_value, max_value,
    COUT: tl.constexpr, G: tl.constexpr, CPG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    n = pid0 // COUT
    c = pid0 % COUT
    g = c // CPG
    mean = tl.load(mean_ptr + n * G + g)
    rstd = tl.load(rstd_ptr + n * G + g)
    gamma = tl.load(gamma_ptr + c).to(tl.float32)
    beta = tl.load(beta_ptr + c).to(tl.float32)
    s = pid1 * BLOCK + tl.arange(0, BLOCK)
    m = s < S
    off = pid0.to(tl.int64) * S + s
    v = tl.load(y_ptr + off, mask=m, other=0.0)
    v = (v - mean) * rstd * gamma + beta
    v = tl.minimum(v, min_value)
    v = tl.maximum(v, min_value)
    v = tl.minimum(v, max_value)
    tl.store(out_ptr + off, v.to(out_ptr.dtype.element_ty), mask=m)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout_p)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, CIN, D, H, W = x.shape
        K = self.kernel_size
        COUT = self.out_channels
        G = self.groups
        CPG = COUT // G
        OD, OH, OW = D - K + 1, H - K + 1, W - K + 1
        S = OD * OH * OW
        dev = x.device

        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        gamma = self.norm.weight.contiguous()
        beta = self.norm.bias.contiguous()

        BLOCK = 256
        NB = triton.cdiv(S, BLOCK)
        y = torch.empty((N, COUT, S), device=dev, dtype=torch.float32)
        psum = torch.empty((N, NB, COUT), device=dev, dtype=torch.float32)
        psq = torch.empty((N, NB, COUT), device=dev, dtype=torch.float32)

        conv3d_kernel[(N, NB)](
            x, w, b, y, psum, psq,
            D, H, W, OD, OH, OW, S, NB,
            CIN=CIN, COUT=COUT, COUT_P=triton.next_power_of_2(COUT),
            K=K, BLOCK=BLOCK, num_warps=4,
        )

        mean = torch.empty((N * G,), device=dev, dtype=torch.float32)
        rstd = torch.empty((N * G,), device=dev, dtype=torch.float32)
        gn_stats_kernel[(N * G,)](
            psum, psq, mean, rstd,
            NB, S, float(self.norm.eps),
            COUT=COUT, G=G, CPG=CPG, CPG_P=triton.next_power_of_2(CPG),
            BLOCK_R=128, num_warps=4,
        )

        out = torch.empty((N, COUT, OD, OH, OW), device=dev, dtype=x.dtype)
        BLOCK_A = 1024
        gn_apply_kernel[(N * COUT, triton.cdiv(S, BLOCK_A))](
            y, out, mean, rstd, gamma, beta,
            S, self.min_value, self.max_value,
            COUT=COUT, G=G, CPG=CPG, BLOCK=BLOCK_A, num_warps=4,
        )
        return out
