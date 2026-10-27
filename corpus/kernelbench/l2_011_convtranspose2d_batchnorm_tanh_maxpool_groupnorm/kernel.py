import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _tanh(x):
    # tanh(x) = 1 - 2 / (exp(2x) + 1); saturates correctly for large |x|
    e = tl.exp(2.0 * x)
    return 1.0 - 2.0 / (e + 1.0)


@triton.jit
def convt_bn_tanh_kernel(
    x_ptr, w_ptr, cb_ptr,
    bn_w_ptr, bn_b_ptr, bn_rm_ptr, bn_rv_ptr, bn_eps,
    y_ptr,
    NB, C_IN, H, W, C_OUT, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr, PAD: tl.constexpr, STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    OHW = OH * OW
    M = NB * OHW
    m_mask = offs_m < M
    n_mask = offs_n < C_OUT

    n_idx = offs_m // OHW
    rem = offs_m % OHW
    oh = rem // OW
    ow = rem % OW

    HW = H * W
    x_base = n_idx.to(tl.int64) * C_IN * HW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kh in tl.static_range(KH):
        th = oh + PAD - kh
        th_c = tl.where(th >= 0, th, 0)
        ih = th_c // STRIDE
        vh = (th >= 0) & ((th_c % STRIDE) == 0) & (ih < H)
        for kw in tl.static_range(KW):
            tw = ow + PAD - kw
            tw_c = tl.where(tw >= 0, tw, 0)
            iw = tw_c // STRIDE
            vw = (tw >= 0) & ((tw_c % STRIDE) == 0) & (iw < W)
            valid = m_mask & vh & vw
            sp_off = x_base + ih * W + iw
            for c0 in range(0, C_IN, BLOCK_K):
                ci = c0 + offs_k
                ci_mask = ci < C_IN
                a_ptrs = x_ptr + sp_off[:, None] + ci[None, :].to(tl.int64) * HW
                a = tl.load(a_ptrs, mask=valid[:, None] & ci_mask[None, :], other=0.0)
                # weight layout: [C_IN, C_OUT, KH, KW]
                b_ptrs = w_ptr + (ci[:, None] * C_OUT + offs_n[None, :]) * (KH * KW) + (kh * KW + kw)
                b = tl.load(b_ptrs, mask=ci_mask[:, None] & n_mask[None, :], other=0.0)
                acc += tl.dot(a, b, input_precision="ieee")

    # epilogue: conv bias + batchnorm (eval) + tanh
    cb = tl.load(cb_ptr + offs_n, mask=n_mask, other=0.0)
    g = tl.load(bn_w_ptr + offs_n, mask=n_mask, other=1.0)
    bb = tl.load(bn_b_ptr + offs_n, mask=n_mask, other=0.0)
    rm = tl.load(bn_rm_ptr + offs_n, mask=n_mask, other=0.0)
    rv = tl.load(bn_rv_ptr + offs_n, mask=n_mask, other=1.0)
    scale = g / tl.sqrt(rv + bn_eps)
    shift = bb + (cb - rm) * scale

    v = acc * scale[None, :] + shift[None, :]
    v = _tanh(v)

    y_off = (n_idx.to(tl.int64) * C_OUT * OHW + rem)[:, None] + offs_n[None, :].to(tl.int64) * OHW
    tl.store(y_ptr + y_off, v, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _pooled(y_ptr, y_base, e, mask, CPG, PHW, PW, OH, OW):
    c_local = e // PHW
    r = e % PHW
    ph = r // PW
    pw = r % PW
    src = y_base + c_local.to(tl.int64) * (OH * OW) + (2 * ph) * OW + 2 * pw
    v0 = tl.load(y_ptr + src, mask=mask, other=0.0)
    v1 = tl.load(y_ptr + src + 1, mask=mask, other=0.0)
    v2 = tl.load(y_ptr + src + OW, mask=mask, other=0.0)
    v3 = tl.load(y_ptr + src + OW + 1, mask=mask, other=0.0)
    v = tl.maximum(tl.maximum(v0, v1), tl.maximum(v2, v3))
    return v, c_local


@triton.jit
def pool_groupnorm_kernel(
    y_ptr, out_ptr, gn_w_ptr, gn_b_ptr, gn_eps,
    C, OH, OW, PH, PW, G, CPG,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    PHW = PH * PW
    numel = CPG * PHW
    y_base = (n.to(tl.int64) * C + g * CPG) * (OH * OW)
    o_base = (n.to(tl.int64) * C + g * CPG) * PHW

    # pass 1: mean
    s = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, numel, BLOCK):
        e = start + tl.arange(0, BLOCK)
        mask = e < numel
        v, _ = _pooled(y_ptr, y_base, e, mask, CPG, PHW, PW, OH, OW)
        s += tl.where(mask, v, 0.0)
    mean = tl.sum(s, axis=0) / numel

    # pass 2: variance
    s2 = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, numel, BLOCK):
        e = start + tl.arange(0, BLOCK)
        mask = e < numel
        v, _ = _pooled(y_ptr, y_base, e, mask, CPG, PHW, PW, OH, OW)
        d = tl.where(mask, v - mean, 0.0)
        s2 += d * d
    var = tl.sum(s2, axis=0) / numel
    rstd = 1.0 / tl.sqrt(var + gn_eps)

    # pass 3: normalize + affine
    for start in range(0, numel, BLOCK):
        e = start + tl.arange(0, BLOCK)
        mask = e < numel
        v, c_local = _pooled(y_ptr, y_base, e, mask, CPG, PHW, PW, OH, OW)
        ch = g * CPG + c_local
        gw = tl.load(gn_w_ptr + ch, mask=mask, other=1.0)
        gb = tl.load(gn_b_ptr + ch, mask=mask, other=0.0)
        o = (v - mean) * rstd * gw + gb
        tl.store(out_ptr + o_base + e, o, mask=mask)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = x.contiguous().to(torch.float32)
        conv = self.conv_transpose
        w = conv.weight.contiguous()
        b = conv.bias.contiguous()
        NB, C_IN, H, W = x.shape
        C_OUT = w.shape[1]
        KH, KW = w.shape[2], w.shape[3]
        S = conv.stride[0]
        PAD = conv.padding[0]
        OH = (H - 1) * S - 2 * PAD + KH
        OW = (W - 1) * S - 2 * PAD + KW

        y = torch.empty((NB, C_OUT, OH, OW), device=x.device, dtype=torch.float32)
        bn = self.batch_norm
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        M = NB * OH * OW
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(C_OUT, BLOCK_N))
        convt_bn_tanh_kernel[grid](
            x, w, b,
            bn.weight, bn.bias, bn.running_mean, bn.running_var, float(bn.eps),
            y,
            NB, C_IN, H, W, C_OUT, OH, OW,
            KH=KH, KW=KW, PAD=PAD, STRIDE=S,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )

        PH, PW = OH // 2, OW // 2
        gn = self.group_norm
        G = gn.num_groups
        CPG = C_OUT // G
        out = torch.empty((NB, C_OUT, PH, PW), device=x.device, dtype=torch.float32)
        pool_groupnorm_kernel[(NB * G,)](
            y, out, gn.weight, gn.bias, float(gn.eps),
            C_OUT, OH, OW, PH, PW, G, CPG,
            BLOCK=1024, num_warps=4,
        )
        return out
