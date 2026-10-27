import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(x_ptr, w_ptr, b_ptr, y_ptr,
                  Cin, H, W, Cout, OH, OW,
                  KH: tl.constexpr, KW: tl.constexpr,
                  BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr):
    n = tl.program_id(0)
    pid_p = tl.program_id(1)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_c = tl.arange(0, BLOCK_C)
    P = OH * OW
    mask_p = offs_p < P
    mask_c = offs_c < Cout
    oh = offs_p // OW
    ow = offs_p % OW

    acc = tl.zeros((BLOCK_P, BLOCK_C), dtype=tl.float32)
    x_base = x_ptr + n.to(tl.int64) * Cin * H * W
    for ci in range(Cin):
        for kh in tl.static_range(KH):
            for kw in tl.static_range(KW):
                xv = tl.load(x_base + ci * H * W + (oh + kh) * W + (ow + kw),
                             mask=mask_p, other=0.0).to(tl.float32)
                wv = tl.load(w_ptr + offs_c * (Cin * KH * KW) + ci * (KH * KW) + kh * KW + kw,
                             mask=mask_c, other=0.0).to(tl.float32)
                acc += xv[:, None] * wv[None, :]
    bv = tl.load(b_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    acc += bv[None, :]

    y_base = y_ptr + n.to(tl.int64) * Cout * P
    ptrs = y_base + offs_c[None, :] * P + offs_p[:, None]
    tl.store(ptrs, acc, mask=mask_p[:, None] & mask_c[None, :])


@triton.jit
def gn_stats_kernel(y_ptr, mean_ptr, rstd_ptr, C, P, G, CPG, eps,
                    BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G
    base = y_ptr + n.to(tl.int64) * C * P + g.to(tl.int64) * CPG * P
    L = CPG * P
    offs = tl.arange(0, BLOCK)
    s = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, L, BLOCK):
        idx = start + offs
        v = tl.load(base + idx, mask=idx < L, other=0.0)
        s += v
    mean = tl.sum(s, axis=0) / L
    s2 = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, L, BLOCK):
        idx = start + offs
        m = idx < L
        v = tl.load(base + idx, mask=m, other=0.0)
        d = tl.where(m, v - mean, 0.0)
        s2 += d * d
    var = tl.sum(s2, axis=0) / L
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)


@triton.jit
def fused_post_kernel(y_ptr, mean_ptr, rstd_ptr, gamma_ptr, beta_ptr, out_ptr,
                      C, P, G, CPG,
                      BLOCK_C: tl.constexpr, BLOCK_P: tl.constexpr):
    n = tl.program_id(0)
    pid_p = tl.program_id(1)
    offs_c = tl.arange(0, BLOCK_C)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    mask_c = offs_c < C
    mask_p = offs_p < P
    mask = mask_c[:, None] & mask_p[None, :]

    y_base = y_ptr + n.to(tl.int64) * C * P
    v = tl.load(y_base + offs_c[:, None] * P + offs_p[None, :], mask=mask, other=0.0)

    grp = offs_c // CPG
    mean = tl.load(mean_ptr + n * G + grp, mask=mask_c, other=0.0)
    rstd = tl.load(rstd_ptr + n * G + grp, mask=mask_c, other=0.0)
    gamma = tl.load(gamma_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)

    xn = (v - mean[:, None]) * rstd[:, None] * gamma[:, None] + beta[:, None]
    # tanh
    xc = tl.minimum(tl.maximum(xn, -20.0), 20.0)
    e = tl.exp(2.0 * xc)
    t = 1.0 - 2.0 / (e + 1.0)
    # hardswish
    hs = t * tl.minimum(tl.maximum(t + 3.0, 0.0), 6.0) / 6.0
    r = v + hs
    r = tl.where(mask, r, float("-inf"))

    mx = tl.max(r, axis=0)
    mx_safe = tl.where(mask_p, mx, 0.0)
    ex = tl.exp(r - mx_safe[None, :])
    ex = tl.where(mask, ex, 0.0)
    ssum = tl.sum(ex, axis=0)
    res = mx_safe + tl.log(ssum)

    out_base = out_ptr + n.to(tl.int64) * P
    tl.store(out_base + offs_p, res.to(out_ptr.dtype.element_ty), mask=mask_p)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()

    def forward(self, x):
        x = x.contiguous()
        N, Cin, H, W = x.shape
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        Cout = w.shape[0]
        KH, KW = w.shape[2], w.shape[3]
        OH = H - KH + 1
        OW = W - KW + 1
        P = OH * OW

        y = torch.empty((N, Cout, OH, OW), device=x.device, dtype=torch.float32)
        BLOCK_C = triton.next_power_of_2(Cout)
        BLOCK_P = 64
        grid = (N, triton.cdiv(P, BLOCK_P))
        conv2d_kernel[grid](x, w, b, y, Cin, H, W, Cout, OH, OW,
                            KH=KH, KW=KW, BLOCK_P=BLOCK_P, BLOCK_C=BLOCK_C)

        G = self.group_norm.num_groups
        CPG = Cout // G
        mean = torch.empty((N * G,), device=x.device, dtype=torch.float32)
        rstd = torch.empty((N * G,), device=x.device, dtype=torch.float32)
        gn_stats_kernel[(N * G,)](y, mean, rstd, Cout, P, G, CPG,
                                  float(self.group_norm.eps), BLOCK=1024)

        out = torch.empty((N, 1, OH, OW), device=x.device, dtype=x.dtype)
        BLOCK_P2 = 64
        grid2 = (N, triton.cdiv(P, BLOCK_P2))
        fused_post_kernel[grid2](y, mean, rstd,
                                 self.group_norm.weight.contiguous(),
                                 self.group_norm.bias.contiguous(),
                                 out, Cout, P, G, CPG,
                                 BLOCK_C=BLOCK_C, BLOCK_P=BLOCK_P2)
        return out
