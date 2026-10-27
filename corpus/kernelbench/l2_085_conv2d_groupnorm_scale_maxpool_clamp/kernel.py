import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    C, H, W, OC, OH, OW,
    K: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_P: tl.constexpr,
):
    n = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_oc = tl.program_id(2)

    P = OH * OW
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_p = offs_p < P
    mask_oc = offs_oc < OC
    oh = offs_p // OW
    ow = offs_p % OW

    acc = tl.zeros((BLOCK_OC, BLOCK_P), dtype=tl.float32)
    x_base = x_ptr + n.to(tl.int64) * C * H * W
    for ic in range(C):
        x_c = x_base + ic * H * W
        w_c = w_ptr + offs_oc * (C * K * K) + ic * K * K
        for kh in tl.static_range(K):
            for kw in tl.static_range(K):
                xv = tl.load(x_c + (oh + kh) * W + (ow + kw), mask=mask_p, other=0.0).to(tl.float32)
                wv = tl.load(w_c + kh * K + kw, mask=mask_oc, other=0.0).to(tl.float32)
                acc += wv[:, None] * xv[None, :]

    bv = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0).to(tl.float32)
    acc += bv[:, None]

    y_base = y_ptr + n.to(tl.int64) * OC * P
    y_ptrs = y_base + offs_oc[:, None] * P + offs_p[None, :]
    tl.store(y_ptrs, acc, mask=mask_oc[:, None] & mask_p[None, :])


@triton.jit
def gn_stats_kernel(
    y_ptr, mean_ptr, rstd_ptr,
    group_size, eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # n * G + g
    base = y_ptr + pid.to(tl.int64) * group_size
    offs = tl.arange(0, BLOCK)

    s = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, group_size, BLOCK):
        idx = start + offs
        v = tl.load(base + idx, mask=idx < group_size, other=0.0)
        s += v
    mean = tl.sum(s, axis=0) / group_size

    ss = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, group_size, BLOCK):
        idx = start + offs
        m = idx < group_size
        v = tl.load(base + idx, mask=m, other=0.0)
        d = tl.where(m, v - mean, 0.0)
        ss += d * d
    var = tl.sum(ss, axis=0) / group_size
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)


@triton.jit
def norm_pool_clamp_kernel(
    y_ptr, mean_ptr, rstd_ptr, gamma_ptr, beta_ptr, scale_ptr, out_ptr,
    OC, G, CPG, OH, OW, PH, PW,
    cmin, cmax,
    PK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_b = tl.program_id(1)
    n = pid_nc // OC
    c = pid_nc % OC
    g = c // CPG

    mean = tl.load(mean_ptr + n * G + g)
    rstd = tl.load(rstd_ptr + n * G + g)
    gamma = tl.load(gamma_ptr + c).to(tl.float32)
    beta = tl.load(beta_ptr + c).to(tl.float32)
    sc = tl.load(scale_ptr + c).to(tl.float32)

    offs = pid_b * BLOCK + tl.arange(0, BLOCK)
    mask = offs < PH * PW
    ph = offs // PW
    pw = offs % PW

    y_base = y_ptr + pid_nc.to(tl.int64) * OH * OW
    best = tl.full((BLOCK,), float("-inf"), dtype=tl.float32)
    for i in tl.static_range(PK):
        for j in tl.static_range(PK):
            r = ph * PK + i
            q = pw * PK + j
            v = tl.load(y_base + r * OW + q, mask=mask, other=0.0)
            v = ((v - mean) * rstd * gamma + beta) * sc
            best = tl.maximum(best, v)

    best = tl.minimum(tl.maximum(best, cmin), cmax)
    out_base = out_ptr + pid_nc.to(tl.int64) * PH * PW
    tl.store(out_base + offs, best.to(out_ptr.dtype.element_ty), mask=mask)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape,
                 maxpool_kernel_size, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.pk = maxpool_kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, C, H, W = x.shape
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        OC = w.shape[0]
        K = w.shape[2]
        OH = H - K + 1
        OW = W - K + 1
        P = OH * OW

        y = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)
        BLOCK_OC = 16
        BLOCK_P = 128
        grid = (N, triton.cdiv(P, BLOCK_P), triton.cdiv(OC, BLOCK_OC))
        conv2d_kernel[grid](
            x, w, b, y,
            C, H, W, OC, OH, OW,
            K=K, BLOCK_OC=BLOCK_OC, BLOCK_P=BLOCK_P,
        )

        G = self.group_norm.num_groups
        CPG = OC // G
        group_size = CPG * P
        mean = torch.empty((N * G,), device=x.device, dtype=torch.float32)
        rstd = torch.empty((N * G,), device=x.device, dtype=torch.float32)
        gn_stats_kernel[(N * G,)](
            y, mean, rstd, group_size, float(self.group_norm.eps),
            BLOCK=1024,
        )

        PK = self.pk
        PH = OH // PK
        PW = OW // PK
        out = torch.empty((N, OC, PH, PW), device=x.device, dtype=x.dtype)
        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()
        scale = self.scale.contiguous().view(-1)
        BLOCK = 256
        grid = (N * OC, triton.cdiv(PH * PW, BLOCK))
        norm_pool_clamp_kernel[grid](
            y, mean, rstd, gamma, beta, scale, out,
            OC, G, CPG, OH, OW, PH, PW,
            float(self.clamp_min), float(self.clamp_max),
            PK=PK, BLOCK=BLOCK,
        )
        return out
