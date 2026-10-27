import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def convt_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, CIN, IH, IW, COUT, OH, OW,
    add_value, scale,
    K: tl.constexpr, STRIDE: tl.constexpr, PAD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_CO: tl.constexpr, BLOCK_CI: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_co = tl.program_id(1)

    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    total_p = N * OH * OW
    p_mask = offs_p < total_p
    co_mask = offs_co < COUT

    ohw = OH * OW
    n = offs_p // ohw
    rem = offs_p % ohw
    oh = rem // OW
    ow = rem % OW
    n64 = n.to(tl.int64)

    acc = tl.zeros((BLOCK_P, BLOCK_CO), dtype=tl.float32)
    ihw = IH * IW
    KK: tl.constexpr = K * K

    for kh in tl.static_range(K):
        th = oh + PAD - kh
        th_c = tl.where(th >= 0, th, 0)
        ih = th_c // STRIDE
        vh = (th >= 0) & ((th_c % STRIDE) == 0) & (ih < IH)
        for kw in tl.static_range(K):
            tw = ow + PAD - kw
            tw_c = tl.where(tw >= 0, tw, 0)
            iw = tw_c // STRIDE
            vw = (tw >= 0) & ((tw_c % STRIDE) == 0) & (iw < IW)
            valid = vh & vw & p_mask
            pix_off = n64 * CIN * ihw + (ih * IW + iw).to(tl.int64)
            for ci0 in range(0, CIN, BLOCK_CI):
                offs_ci = ci0 + tl.arange(0, BLOCK_CI)
                ci_mask = offs_ci < CIN
                x_ptrs = x_ptr + pix_off[:, None] + (offs_ci.to(tl.int64) * ihw)[None, :]
                xm = valid[:, None] & ci_mask[None, :]
                a = tl.load(x_ptrs, mask=xm, other=0.0)
                w_ptrs = w_ptr + (offs_ci * (COUT * KK))[:, None] + (offs_co * KK)[None, :] + (kh * K + kw)
                wm = ci_mask[:, None] & co_mask[None, :]
                b = tl.load(w_ptrs, mask=wm, other=0.0)
                b = b.to(a.dtype)
                acc = tl.dot(a, b, acc)

    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_co, mask=co_mask, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    # Mish: y * tanh(softplus(y)), softplus threshold 20 (PyTorch semantics)
    y = acc
    sp = tl.where(y > 20.0, y, tl.log(1.0 + tl.exp(tl.minimum(y, 20.0))))
    e = tl.exp(-2.0 * sp)
    t = (1.0 - e) / (1.0 + e)
    y = y * t
    y = y + add_value
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)
    y = y * scale

    out_off = (n64 * COUT * ohw + (oh * OW + ow).to(tl.int64))[:, None] + \
        (offs_co.to(tl.int64) * ohw)[None, :]
    om = p_mask[:, None] & co_mask[None, :]
    tl.store(out_ptr + out_off, y.to(out_ptr.dtype.element_ty), mask=om)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        conv = self.conv_transpose
        x = x.contiguous()
        w = conv.weight.contiguous()
        b = conv.bias
        N, CIN, IH, IW = x.shape
        COUT = w.shape[1]
        K = conv.kernel_size[0]
        S = conv.stride[0]
        P = conv.padding[0]
        OP = conv.output_padding[0]
        OH = (IH - 1) * S - 2 * P + (K - 1) + OP + 1
        OW = (IW - 1) * S - 2 * P + (K - 1) + OP + 1
        out = torch.empty((N, COUT, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_P = 128
        BLOCK_CO = 64
        BLOCK_CI = 32
        has_bias = b is not None
        b_arg = b.contiguous() if has_bias else w
        grid = (triton.cdiv(N * OH * OW, BLOCK_P), triton.cdiv(COUT, BLOCK_CO))
        convt_mish_kernel[grid](
            x, w, b_arg, out,
            N, CIN, IH, IW, COUT, OH, OW,
            float(self.add_value), float(self.scale),
            K=K, STRIDE=S, PAD=P, HAS_BIAS=has_bias,
            BLOCK_P=BLOCK_P, BLOCK_CO=BLOCK_CO, BLOCK_CI=BLOCK_CI,
            num_warps=4,
        )
        return out
