import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _masked_tap_sums_kernel(
    x_ptr, s_ptr,
    H, W, Ho, Wo,
    STRIDE: tl.constexpr, PAD: tl.constexpr, DIL: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr, KK_PAD: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)  # index over N*IC
    base = x_ptr + pid.to(tl.int64) * H * W

    offs_h = tl.arange(0, BLOCK_H)
    offs_w = tl.arange(0, BLOCK_W)
    kk_idx = tl.arange(0, KK_PAD)
    acc = tl.zeros([KK_PAD], dtype=tl.float32)

    for h0 in range(0, H, BLOCK_H):
        ih = h0 + offs_h
        hmask = ih < H
        for w0 in range(0, W, BLOCK_W):
            iw = w0 + offs_w
            wmask = iw < W
            m = hmask[:, None] & wmask[None, :]
            ptrs = base + ih[:, None] * W + iw[None, :]
            xv = tl.load(ptrs, mask=m, other=0.0).to(tl.float32)
            for kh in tl.static_range(KH):
                oh = ih * STRIDE - PAD + kh * DIL
                rv = (oh >= 0) & (oh < Ho) & hmask
                # row-masked column sums
                colsum = tl.sum(tl.where(rv[:, None], xv, 0.0), axis=0)  # [BLOCK_W]
                for kw in tl.static_range(KW):
                    ow = iw * STRIDE - PAD + kw * DIL
                    cv = (ow >= 0) & (ow < Wo) & wmask
                    s = tl.sum(tl.where(cv, colsum, 0.0), axis=0)
                    acc += tl.where(kk_idx == kh * KW + kw, s, 0.0)

    tl.store(s_ptr + pid.to(tl.int64) * (KH * KW) + kk_idx, acc, mask=kk_idx < KH * KW)


@triton.jit
def _reduce_kernel(
    s_ptr, w_ptr, b_ptr, out_ptr,
    OC, J, scale_mean, multiplier,
    KK: tl.constexpr, HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_J: tl.constexpr,
):
    n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    ocmask = oc < OC
    offs_j = tl.arange(0, BLOCK_J)

    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)
    for j0 in range(0, J, BLOCK_J):
        j = j0 + offs_j
        jmask = j < J
        sv = tl.load(s_ptr + n * J + j, mask=jmask, other=0.0)
        ic = j // KK
        r = j % KK
        w_off = ic[:, None] * (OC * KK) + oc[None, :] * KK + r[:, None]
        wm = jmask[:, None] & ocmask[None, :]
        wv = tl.load(w_ptr + w_off, mask=wm, other=0.0).to(tl.float32)
        acc += tl.sum(sv[:, None] * wv, axis=0)

    res = acc * scale_mean
    if HAS_BIAS:
        bv = tl.load(b_ptr + oc, mask=ocmask, other=0.0).to(tl.float32)
        res = res + bv
    res = res * multiplier
    tl.store(out_ptr + n * OC + oc, res.to(out_ptr.dtype.element_ty), mask=ocmask)


def _pair(v):
    if isinstance(v, (tuple, list)):
        return tuple(v)
    return (v, v)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding,
                                                 output_padding=output_padding)
        self.multiplier = multiplier

    def forward(self, x):
        conv = self.conv_transpose
        N, IC, H, W = x.shape
        OC = conv.out_channels
        KH, KW = _pair(conv.kernel_size)
        sh, sw = _pair(conv.stride)
        ph, pw = _pair(conv.padding)
        oph, opw = _pair(conv.output_padding)
        dh, dw = _pair(conv.dilation)
        assert sh == sw and ph == pw and dh == dw and conv.groups == 1
        Ho = (H - 1) * sh - 2 * ph + dh * (KH - 1) + oph + 1
        Wo = (W - 1) * sw - 2 * pw + dw * (KW - 1) + opw + 1

        x = x.contiguous()
        KK = KH * KW
        KK_PAD = triton.next_power_of_2(KK)
        S = torch.empty((N * IC * KK,), device=x.device, dtype=torch.float32)

        BLOCK_W = min(128, triton.next_power_of_2(W))
        BLOCK_H = max(1, min(triton.next_power_of_2(H), 4096 // BLOCK_W))
        _masked_tap_sums_kernel[(N * IC,)](
            x, S, H, W, Ho, Wo,
            STRIDE=sh, PAD=ph, DIL=dh, KH=KH, KW=KW, KK_PAD=KK_PAD,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, num_warps=4,
        )

        out = torch.empty((N, OC, 1, 1), device=x.device, dtype=x.dtype)
        weight = conv.weight.contiguous()
        has_bias = conv.bias is not None
        bias = conv.bias if has_bias else weight
        J = IC * KK
        BLOCK_OC = 64
        BLOCK_J = 64
        grid = (N, triton.cdiv(OC, BLOCK_OC))
        _reduce_kernel[grid](
            S, weight, bias, out,
            OC, J, 1.0 / float(Ho * Wo), float(self.multiplier),
            KK=KK, HAS_BIAS=has_bias,
            BLOCK_OC=BLOCK_OC, BLOCK_J=BLOCK_J, num_warps=4,
        )
        return out
