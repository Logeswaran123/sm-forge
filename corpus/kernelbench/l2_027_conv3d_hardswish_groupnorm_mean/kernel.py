import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_hs_stats_kernel(
    x_ptr, w_ptr, bias_ptr, psum_ptr, psq_ptr,
    C_in, D, H, W, OD, OH, OW, S, NB, C_out,
    K: tl.constexpr, HAS_BIAS: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr,
):
    b = tl.program_id(0)
    pid = tl.program_id(1)

    offs = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    smask = offs < S
    ohw = OH * OW
    od = offs // ohw
    rem = offs % ohw
    oh = rem // OW
    ow = rem % OW

    co = tl.arange(0, BLOCK_C)
    cmask = co < C_out

    HW = H * W
    DHW = D * HW
    x_base = x_ptr + b.to(tl.int64) * C_in * DHW
    sp = od * HW + oh * W + ow
    sp = tl.where(smask, sp, 0)

    KKK: tl.constexpr = K * K * K
    R = C_in * KKK

    acc = tl.zeros((BLOCK_S, BLOCK_C), dtype=tl.float32)
    for ci in range(C_in):
        x_ci = x_base + ci * DHW
        w_ci = w_ptr + co * R + ci * KKK
        for kd in tl.static_range(K):
            for kh in tl.static_range(K):
                for kw in tl.static_range(K):
                    xv = tl.load(x_ci + sp + kd * HW + kh * W + kw,
                                 mask=smask, other=0.0).to(tl.float32)
                    wv = tl.load(w_ci + (kd * K * K + kh * K + kw),
                                 mask=cmask, other=0.0).to(tl.float32)
                    acc += xv[:, None] * wv[None, :]

    if HAS_BIAS:
        bv = tl.load(bias_ptr + co, mask=cmask, other=0.0).to(tl.float32)
        acc += bv[None, :]

    # HardSwish: x * relu6(x + 3) / 6
    t = tl.minimum(tl.maximum(acc + 3.0, 0.0), 6.0)
    h = acc * t / 6.0
    h = tl.where(smask[:, None], h, 0.0)

    s1 = tl.sum(h, axis=0)
    s2 = tl.sum(h * h, axis=0)

    out_off = (b * NB + pid) * C_out + co
    tl.store(psum_ptr + out_off, s1, mask=cmask)
    tl.store(psq_ptr + out_off, s2, mask=cmask)


@triton.jit
def gn_finalize_kernel(
    psum_ptr, psq_ptr, gamma_ptr, beta_ptr, out_ptr,
    NB, C_out, CPG, S, eps,
    BLOCK_NB: tl.constexpr, BLOCK_C: tl.constexpr,
):
    b = tl.program_id(0)
    co = tl.arange(0, BLOCK_C)
    cmask = co < C_out

    s1 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    s2 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for start in range(0, NB, BLOCK_NB):
        nb = start + tl.arange(0, BLOCK_NB)
        nmask = nb < NB
        offs = (b * NB + nb)[:, None] * C_out + co[None, :]
        m = nmask[:, None] & cmask[None, :]
        p1 = tl.load(psum_ptr + offs, mask=m, other=0.0)
        p2 = tl.load(psq_ptr + offs, mask=m, other=0.0)
        s1 += tl.sum(p1, axis=0)
        s2 += tl.sum(p2, axis=0)

    # group sums via same-group mask matrix
    co2 = tl.arange(0, BLOCK_C)
    g_row = co // CPG
    g_col = co2 // CPG
    same = (g_row[:, None] == g_col[None, :]) & cmask[:, None] & (co2 < C_out)[None, :]
    samef = same.to(tl.float32)
    gs1 = tl.sum(samef * s1[None, :], axis=1)
    gs2 = tl.sum(samef * s2[None, :], axis=1)

    n = (CPG * S).to(tl.float32)
    mu = gs1 / n
    var = gs2 / n - mu * mu
    var = tl.maximum(var, 0.0)
    rstd = 1.0 / tl.sqrt(var + eps)

    cmean = s1 / S.to(tl.float32)
    gamma = tl.load(gamma_ptr + co, mask=cmask, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + co, mask=cmask, other=0.0).to(tl.float32)
    out = gamma * (cmean - mu) * rstd + beta
    tl.store(out_ptr + b * C_out + co, out.to(out_ptr.dtype.element_ty), mask=cmask)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = x.contiguous()
        B, C_in, D, H, W = x.shape
        w = self.conv.weight.contiguous()
        C_out = w.shape[0]
        K = w.shape[2]
        OD, OH, OW = D - K + 1, H - K + 1, W - K + 1
        S = OD * OH * OW

        BLOCK_S = 256
        BLOCK_C = triton.next_power_of_2(C_out)
        NB = triton.cdiv(S, BLOCK_S)

        psum = torch.empty((B, NB, C_out), device=x.device, dtype=torch.float32)
        psq = torch.empty((B, NB, C_out), device=x.device, dtype=torch.float32)

        has_bias = self.conv.bias is not None
        bias = self.conv.bias if has_bias else w

        conv_hs_stats_kernel[(B, NB)](
            x, w, bias, psum, psq,
            C_in, D, H, W, OD, OH, OW, S, NB, C_out,
            K=K, HAS_BIAS=has_bias, BLOCK_S=BLOCK_S, BLOCK_C=BLOCK_C,
            num_warps=4,
        )

        G = self.group_norm.num_groups
        CPG = C_out // G
        out = torch.empty((B, C_out), device=x.device, dtype=x.dtype)
        gn_finalize_kernel[(B,)](
            psum, psq, self.group_norm.weight, self.group_norm.bias, out,
            NB, C_out, CPG, S, float(self.group_norm.eps),
            BLOCK_NB=64, BLOCK_C=BLOCK_C,
            num_warps=4,
        )
        return out
