import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def convt_swish_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    CIN, COUT, D, H, W, OD, OH, OW,
    STRIDE: tl.constexpr, PAD: tl.constexpr, K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_s = tl.program_id(0)
    n = tl.program_id(1).to(tl.int64)

    S = OD * OH * OW
    offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs < S
    ow = offs % OW
    t = offs // OW
    oh = t % OH
    od = t // OH

    co = tl.arange(0, BLOCK_C)
    mask_c = co < COUT

    acc = tl.zeros((BLOCK_S, BLOCK_C), dtype=tl.float32)
    HW = H * W
    for ci in range(CIN):
        x_base = x_ptr + (n * CIN + ci) * D * HW
        for kd in tl.static_range(K):
            td = od + PAD - kd
            idd = td // STRIDE
            vd = (td >= 0) & ((td % STRIDE) == 0) & (idd < D)
            for kh in tl.static_range(K):
                th = oh + PAD - kh
                ih = th // STRIDE
                vh = (th >= 0) & ((th % STRIDE) == 0) & (ih < H)
                for kw in tl.static_range(K):
                    tw = ow + PAD - kw
                    iw = tw // STRIDE
                    vw = (tw >= 0) & ((tw % STRIDE) == 0) & (iw < W)
                    valid = mask_s & vd & vh & vw
                    xv = tl.load(x_base + idd * HW + ih * W + iw, mask=valid, other=0.0).to(tl.float32)
                    w_off = ((ci * COUT + co) * K + kd) * (K * K) + kh * K + kw
                    wv = tl.load(w_ptr + w_off, mask=mask_c, other=0.0).to(tl.float32)
                    acc += xv[:, None] * wv[None, :]

    if HAS_BIAS:
        bv = tl.load(b_ptr + co, mask=mask_c, other=0.0).to(tl.float32)
        acc += bv[None, :]

    # Swish
    acc = acc / (1.0 + tl.exp(-acc))

    y_off = (n * COUT + co[None, :].to(tl.int64)) * S + offs[:, None]
    tl.store(y_ptr + y_off, acc, mask=mask_s[:, None] & mask_c[None, :])


@triton.jit
def gn_stats_kernel(y_ptr, mean_ptr, rstd_ptr, count, eps, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = y_ptr + pid.to(tl.int64) * count
    s = tl.zeros((BLOCK,), dtype=tl.float32)
    ss = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, count, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < count
        v = tl.load(base + offs, mask=m, other=0.0)
        s += v
        ss += v * v
    tot = tl.sum(s, axis=0)
    tot2 = tl.sum(ss, axis=0)
    mean = tot / count
    var = tot2 / count - mean * mean
    var = tl.maximum(var, 0.0)
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)


@triton.jit
def gn_apply_hswish_kernel(
    y_ptr, out_ptr, mean_ptr, rstd_ptr, g_ptr, b_ptr,
    S, COUT, CPG, G,
    BLOCK: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_nc = tl.program_id(1)
    c = pid_nc % COUT
    n = pid_nc // COUT
    grp = c // CPG
    mean = tl.load(mean_ptr + n * G + grp)
    rstd = tl.load(rstd_ptr + n * G + grp)
    gamma = tl.load(g_ptr + c).to(tl.float32)
    beta = tl.load(b_ptr + c).to(tl.float32)

    offs = pid_s * BLOCK + tl.arange(0, BLOCK)
    m = offs < S
    base = pid_nc.to(tl.int64) * S
    v = tl.load(y_ptr + base + offs, mask=m, other=0.0)
    v = (v - mean) * rstd * gamma + beta
    r = tl.minimum(tl.maximum(v + 3.0, 0.0), 6.0)
    v = v * r / 6.0
    tl.store(out_ptr + base + offs, v.to(out_ptr.dtype.element_ty), mask=m)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)

    def forward(self, x):
        x = x.contiguous()
        conv = self.conv_transpose
        w = conv.weight.contiguous()
        b = conv.bias
        K = conv.kernel_size[0]
        stride = conv.stride[0]
        pad = conv.padding[0]
        outpad = conv.output_padding[0]

        N, CIN, D, H, W = x.shape
        COUT = w.shape[1]
        OD = (D - 1) * stride - 2 * pad + K + outpad
        OH = (H - 1) * stride - 2 * pad + K + outpad
        OW = (W - 1) * stride - 2 * pad + K + outpad
        S = OD * OH * OW

        y = torch.empty((N, COUT, OD, OH, OW), device=x.device, dtype=torch.float32)
        BLOCK_S = 128
        BLOCK_C = triton.next_power_of_2(COUT)
        has_bias = b is not None
        b_arg = b.contiguous() if has_bias else w
        grid = (triton.cdiv(S, BLOCK_S), N)
        convt_swish_kernel[grid](
            x, w, b_arg, y,
            CIN, COUT, D, H, W, OD, OH, OW,
            STRIDE=stride, PAD=pad, K=K, HAS_BIAS=has_bias,
            BLOCK_S=BLOCK_S, BLOCK_C=BLOCK_C,
        )

        gn = self.group_norm
        G = gn.num_groups
        CPG = COUT // G
        count = CPG * S
        mean = torch.empty((N * G,), device=x.device, dtype=torch.float32)
        rstd = torch.empty((N * G,), device=x.device, dtype=torch.float32)
        gn_stats_kernel[(N * G,)](y, mean, rstd, count, float(gn.eps), BLOCK=2048)

        out = torch.empty((N, COUT, OD, OH, OW), device=x.device, dtype=x.dtype)
        BLOCK = 1024
        grid2 = (triton.cdiv(S, BLOCK), N * COUT)
        gn_apply_hswish_kernel[grid2](
            y, out, mean, rstd, gn.weight.contiguous(), gn.bias.contiguous(),
            S, COUT, CPG, G, BLOCK=BLOCK,
        )
        return out
