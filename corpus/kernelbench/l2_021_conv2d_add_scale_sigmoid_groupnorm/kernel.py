import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_bias_scale_sigmoid_kernel(
    x_ptr, w_ptr, cb_ptr, b_ptr, s_ptr, y_ptr,
    C_in, H, W, C_out, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0).to(tl.int64)
    pb = tl.program_id(1)
    P = OH * OW
    offs_p = pb * BLOCK_P + tl.arange(0, BLOCK_P)
    mask_p = offs_p < P
    oh = offs_p // OW
    ow = offs_p % OW
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C_out

    acc = tl.zeros([BLOCK_P, BLOCK_C], dtype=tl.float32)
    for ic in range(C_in):
        x_base = x_ptr + (n * C_in + ic) * H * W
        for kh in tl.static_range(KH):
            for kw in tl.static_range(KW):
                xv = tl.load(x_base + (oh + kh) * W + (ow + kw), mask=mask_p, other=0.0).to(tl.float32)
                wv = tl.load(w_ptr + ((offs_c * C_in + ic) * KH + kh) * KW + kw,
                             mask=mask_c, other=0.0).to(tl.float32)
                acc += xv[:, None] * wv[None, :]

    cb = tl.load(cb_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    bb = tl.load(b_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    ss = tl.load(s_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    v = (acc + cb[None, :] + bb[None, :]) * ss[None, :]
    v = 1.0 / (1.0 + tl.exp(-v))

    y_offs = (n * C_out + offs_c[None, :]) * P + offs_p[:, None]
    tl.store(y_ptr + y_offs, v, mask=mask_p[:, None] & mask_c[None, :])


@triton.jit
def group_stats_kernel(y_ptr, mean_ptr, rstd_ptr, L, eps, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = y_ptr + pid.to(tl.int64) * L
    acc_s = tl.zeros([BLOCK], dtype=tl.float32)
    acc_q = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < L
        v = tl.load(base + offs, mask=m, other=0.0)
        acc_s += v
        acc_q += v * v
    s = tl.sum(acc_s, axis=0)
    q = tl.sum(acc_q, axis=0)
    mean = s / L
    var = q / L - mean * mean
    var = tl.maximum(var, 0.0)
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)


@triton.jit
def group_norm_apply_kernel(
    y_ptr, out_ptr, mean_ptr, rstd_ptr, gw_ptr, gb_ptr,
    C, P, CPG, G,
    BLOCK: tl.constexpr,
):
    nc = tl.program_id(0)
    pb = tl.program_id(1)
    n = nc // C
    c = nc % C
    g = c // CPG
    mean = tl.load(mean_ptr + n * G + g)
    rstd = tl.load(rstd_ptr + n * G + g)
    gw = tl.load(gw_ptr + c).to(tl.float32)
    gb = tl.load(gb_ptr + c).to(tl.float32)
    offs = pb * BLOCK + tl.arange(0, BLOCK)
    m = offs < P
    base = nc.to(tl.int64) * P
    v = tl.load(y_ptr + base + offs, mask=m, other=0.0)
    o = (v - mean) * rstd * gw + gb
    tl.store(out_ptr + base + offs, o.to(out_ptr.dtype.element_ty), mask=m)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = x.contiguous()
        N, C_in, H, W = x.shape
        w = self.conv.weight.contiguous()
        C_out, _, KH, KW = w.shape
        OH = H - KH + 1
        OW = W - KW + 1
        P = OH * OW
        dev = x.device

        cb = self.conv.bias.contiguous()
        bb = self.bias.reshape(-1).contiguous()
        ss = self.scale.reshape(-1).contiguous()

        y = torch.empty((N, C_out, OH, OW), device=dev, dtype=torch.float32)
        BLOCK_P = 128
        BLOCK_C = triton.next_power_of_2(C_out)
        grid = (N, triton.cdiv(P, BLOCK_P))
        conv_bias_scale_sigmoid_kernel[grid](
            x, w, cb, bb, ss, y,
            C_in, H, W, C_out, OH, OW,
            KH=KH, KW=KW, BLOCK_P=BLOCK_P, BLOCK_C=BLOCK_C,
        )

        G = self.group_norm.num_groups
        CPG = C_out // G
        L = CPG * P
        mean = torch.empty((N * G,), device=dev, dtype=torch.float32)
        rstd = torch.empty((N * G,), device=dev, dtype=torch.float32)
        group_stats_kernel[(N * G,)](y, mean, rstd, L, float(self.group_norm.eps), BLOCK=2048)

        out = torch.empty((N, C_out, OH, OW), device=dev, dtype=x.dtype)
        BLOCK = 1024
        group_norm_apply_kernel[(N * C_out, triton.cdiv(P, BLOCK))](
            y, out, mean, rstd,
            self.group_norm.weight.contiguous(), self.group_norm.bias.contiguous(),
            C_out, P, CPG, G, BLOCK=BLOCK,
        )
        return out
