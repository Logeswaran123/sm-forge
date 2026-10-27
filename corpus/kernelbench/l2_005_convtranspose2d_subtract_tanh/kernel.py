import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _convt_bias_tanh_kernel(
    x_ptr, w_ptr, cb_ptr, sb_ptr, y_ptr,
    N, CI, H, W, CO, OH, OW, QH, QW,
    STRIDE: tl.constexpr, PAD: tl.constexpr, KS: tl.constexpr, KT: tl.constexpr,
    HAS_CBIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    par = tl.program_id(1)
    pid_n = tl.program_id(2)

    ph = par // STRIDE
    pw = par % STRIDE

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    M = N * QH * QW
    m_mask = offs_m < M
    nb = offs_m // (QH * QW)
    r = offs_m % (QH * QW)
    a = r // QW
    b = r % QW
    oh = a * STRIDE + ph
    ow = b * STRIDE + pw
    out_mask_m = m_mask & (oh < OH) & (ow < OW)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < CO

    # Kernel taps contributing to this parity class
    kh0 = (ph + PAD) % STRIDE
    kw0 = (pw + PAD) % STRIDE
    bh = (ph + PAD - kh0) // STRIDE
    bw = (pw + PAD - kw0) // STRIDE

    HW = H * W
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for jh in tl.static_range(KT):
        kh = kh0 + jh * STRIDE
        ih = a + bh - jh
        vh = (ih >= 0) & (ih < H)
        for jw in tl.static_range(KT):
            kw = kw0 + jw * STRIDE
            iw = b + bw - jw
            valid = out_mask_m & vh & (iw >= 0) & (iw < W)
            x_base = nb * (CI * HW) + ih * W + iw
            for k0 in range(0, CI, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                k_mask = offs_k < CI
                a_tile = tl.load(
                    x_ptr + x_base[:, None] + offs_k[None, :] * HW,
                    mask=valid[:, None] & k_mask[None, :], other=0.0,
                )
                w_tile = tl.load(
                    w_ptr + offs_k[:, None] * (CO * KS * KS)
                    + offs_n[None, :] * (KS * KS) + kh * KS + kw,
                    mask=k_mask[:, None] & n_mask[None, :], other=0.0,
                )
                acc = tl.dot(a_tile, w_tile, acc)

    sb = tl.load(sb_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    bias = -sb
    if HAS_CBIAS:
        cb = tl.load(cb_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        bias = bias + cb
    v = acc + bias[None, :]

    # tanh via exp, numerically stable
    t = tl.exp(-2.0 * tl.abs(v))
    th = (1.0 - t) / (1.0 + t)
    res = tl.where(v >= 0, th, -th)

    y_off = (nb * (CO * OH * OW) + oh * OW + ow)[:, None] + offs_n[None, :] * (OH * OW)
    tl.store(y_ptr + y_off, res.to(y_ptr.dtype.element_ty),
             mask=out_mask_m[:, None] & n_mask[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                                 padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        conv = self.conv_transpose
        x = x.contiguous()
        N, CI, H, W = x.shape
        w = conv.weight.to(x.dtype).contiguous()  # [CI, CO, K, K]
        CO = w.shape[1]
        KS = w.shape[2]
        STRIDE = conv.stride[0]
        PAD = conv.padding[0]
        OPAD = conv.output_padding[0]
        assert conv.stride[0] == conv.stride[1] and conv.padding[0] == conv.padding[1]
        assert conv.output_padding[0] == conv.output_padding[1]
        assert conv.dilation == (1, 1) and conv.groups == 1
        assert w.shape[2] == w.shape[3] and KS % STRIDE == 0
        KT = KS // STRIDE

        OH = (H - 1) * STRIDE - 2 * PAD + KS + OPAD
        OW = (W - 1) * STRIDE - 2 * PAD + KS + OPAD
        QH = (OH + STRIDE - 1) // STRIDE
        QW = (OW + STRIDE - 1) // STRIDE

        y = torch.empty((N, CO, OH, OW), device=x.device, dtype=x.dtype)
        sb = self.bias.reshape(-1).contiguous()
        has_cb = conv.bias is not None
        cb = conv.bias.contiguous() if has_cb else sb

        BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 32
        M = N * QH * QW
        grid = (triton.cdiv(M, BLOCK_M), STRIDE * STRIDE, triton.cdiv(CO, BLOCK_N))
        _convt_bias_tanh_kernel[grid](
            x, w, cb, sb, y,
            N, CI, H, W, CO, OH, OW, QH, QW,
            STRIDE=STRIDE, PAD=PAD, KS=KS, KT=KT, HAS_CBIAS=has_cb,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )
        return y
