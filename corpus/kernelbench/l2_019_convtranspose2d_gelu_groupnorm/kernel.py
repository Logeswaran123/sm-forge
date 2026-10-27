import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def convT_gelu_stats_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, psum_ptr, psq_ptr,
    Cin, H, W, Cout, OH, OW, P, num_m,
    KS: tl.constexpr, S: tl.constexpr, G: tl.constexpr, CPG: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    n = tl.program_id(1).to(tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < P
    oh = offs_m // OW
    ow = offs_m % OW

    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < Cout

    x_base = x_ptr + n * Cin * H * W
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kh in tl.static_range(KS):
        for kw in tl.static_range(KS):
            ihn = oh - kh
            iwn = ow - kw
            valid = m_mask & (ihn >= 0) & (iwn >= 0)
            ihn_c = tl.where(valid, ihn, 0)
            iwn_c = tl.where(valid, iwn, 0)
            valid = valid & ((ihn_c % S) == 0) & ((iwn_c % S) == 0)
            ih = ihn_c // S
            iw = iwn_c // S
            valid = valid & (ih < H) & (iw < W)
            ih = tl.where(valid, ih, 0)
            iw = tl.where(valid, iw, 0)
            sp = ih * W + iw
            for ci0 in range(0, Cin, BLOCK_K):
                offs_k = ci0 + tl.arange(0, BLOCK_K)
                k_mask = offs_k < Cin
                a_ptrs = x_base + offs_k[None, :] * (H * W) + sp[:, None]
                a = tl.load(a_ptrs, mask=valid[:, None] & k_mask[None, :], other=0.0)
                b_ptrs = (w_ptr + offs_k[:, None] * (Cout * KS * KS)
                          + offs_n[None, :] * (KS * KS) + (kh * KS + kw))
                b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
                acc += tl.dot(a, b.to(a.dtype))

    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    v = acc + bias[None, :]
    v = 0.5 * v * (1.0 + tl.math.erf(v * 0.7071067811865476))

    out_mask = m_mask[:, None] & n_mask[None, :]
    y_ptrs = y_ptr + n * Cout * P + offs_n[None, :] * P + offs_m[:, None]
    tl.store(y_ptrs, v, mask=out_mask)

    gid = offs_n // CPG
    pbase = (n * num_m + pid_m) * G
    for g in tl.static_range(G):
        sel = out_mask & (gid[None, :] == g)
        vv = tl.where(sel, v, 0.0)
        s = tl.sum(tl.sum(vv, axis=1), axis=0)
        sq = tl.sum(tl.sum(vv * vv, axis=1), axis=0)
        tl.store(psum_ptr + pbase + g, s)
        tl.store(psq_ptr + pbase + g, sq)


@triton.jit
def gn_stats_kernel(
    psum_ptr, psq_ptr, mean_ptr, rstd_ptr,
    num_m, count, eps,
    G: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # n * G + g
    n = pid // G
    g = pid % G
    s = tl.zeros((BLOCK,), dtype=tl.float32)
    sq = tl.zeros((BLOCK,), dtype=tl.float32)
    for t0 in range(0, num_m, BLOCK):
        offs = t0 + tl.arange(0, BLOCK)
        msk = offs < num_m
        idx = (n * num_m + offs) * G + g
        s += tl.load(psum_ptr + idx, mask=msk, other=0.0)
        sq += tl.load(psq_ptr + idx, mask=msk, other=0.0)
    tot = tl.sum(s, axis=0)
    totsq = tl.sum(sq, axis=0)
    mean = tot / count
    var = totsq / count - mean * mean
    var = tl.maximum(var, 0.0)
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)


@triton.jit
def gn_apply_kernel(
    y_ptr, out_ptr, mean_ptr, rstd_ptr, gamma_ptr, beta_ptr,
    Cout, P,
    G: tl.constexpr, CPG: tl.constexpr, BLOCK: tl.constexpr,
):
    pid_p = tl.program_id(0)
    row = tl.program_id(1)  # n * Cout + c
    n = row // Cout
    c = row % Cout
    g = c // CPG
    mean = tl.load(mean_ptr + n * G + g)
    rstd = tl.load(rstd_ptr + n * G + g)
    gamma = tl.load(gamma_ptr + c).to(tl.float32)
    beta = tl.load(beta_ptr + c).to(tl.float32)
    offs = pid_p * BLOCK + tl.arange(0, BLOCK)
    msk = offs < P
    base = row.to(tl.int64) * P
    v = tl.load(y_ptr + base + offs, mask=msk, other=0.0)
    o = (v - mean) * rstd * gamma + beta
    tl.store(out_ptr + base + offs, o.to(out_ptr.dtype.element_ty), mask=msk)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = x.contiguous()
        N, Cin, H, W = x.shape
        w = self.conv_transpose.weight.contiguous()
        b = self.conv_transpose.bias.contiguous()
        Cout = w.shape[1]
        KS = w.shape[2]
        S = self.conv_transpose.stride[0]
        OH = (H - 1) * S + KS
        OW = (W - 1) * S + KS
        P = OH * OW
        G = self.group_norm.num_groups
        CPG = Cout // G
        eps = float(self.group_norm.eps)

        BLOCK_M = 64
        BLOCK_N = max(16, triton.next_power_of_2(Cout))
        BLOCK_K = 32
        num_m = triton.cdiv(P, BLOCK_M)

        dev = x.device
        y = torch.empty((N, Cout, OH, OW), device=dev, dtype=torch.float32)
        psum = torch.empty((N, num_m, G), device=dev, dtype=torch.float32)
        psq = torch.empty((N, num_m, G), device=dev, dtype=torch.float32)

        convT_gelu_stats_kernel[(num_m, N)](
            x, w, b, y, psum, psq,
            Cin, H, W, Cout, OH, OW, P, num_m,
            KS=KS, S=S, G=G, CPG=CPG,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )

        mean = torch.empty((N * G,), device=dev, dtype=torch.float32)
        rstd = torch.empty((N * G,), device=dev, dtype=torch.float32)
        count = float(CPG * P)
        gn_stats_kernel[(N * G,)](
            psum, psq, mean, rstd, num_m, count, eps,
            G=G, BLOCK=1024, num_warps=4,
        )

        out = torch.empty((N, Cout, OH, OW), device=dev, dtype=x.dtype)
        BLOCK = 1024
        gn_apply_kernel[(triton.cdiv(P, BLOCK), N * Cout)](
            y, out, mean, rstd,
            self.group_norm.weight, self.group_norm.bias,
            Cout, P,
            G=G, CPG=CPG, BLOCK=BLOCK, num_warps=4,
        )
        return out
