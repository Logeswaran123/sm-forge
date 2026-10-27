import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def convt_softmax_bias_sigmoid_kernel(
    x_ptr, w_ptr, cb_ptr, b_ptr, out_ptr,
    N, C_IN, H, W, C_OUT, OH, OW,
    scale,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    P_total = N * OH * OW
    offs_p = pid * BLOCK_P + tl.arange(0, BLOCK_P)
    pmask = offs_p < P_total
    offs_p_c = tl.where(pmask, offs_p, 0)

    ohw = OH * OW
    n = offs_p_c // ohw
    rem = offs_p_c % ohw
    oh = rem // OW
    ow = rem % OW

    offs_co = tl.arange(0, BLOCK_CO)
    co_mask = offs_co < C_OUT
    offs_k = tl.arange(0, BLOCK_K)

    n64 = n.to(tl.int64)
    x_base = n64 * C_IN * H * W  # [BLOCK_P]

    acc = tl.zeros((BLOCK_P, BLOCK_CO), dtype=tl.float32)

    for kh in tl.static_range(KH):
        ih_num = oh + PAD - kh
        ih_c = tl.where(ih_num >= 0, ih_num, 0)
        vh = (ih_num >= 0) & ((ih_c % STRIDE) == 0)
        ih = ih_c // STRIDE
        vh = vh & (ih < H)
        for kw in tl.static_range(KW):
            iw_num = ow + PAD - kw
            iw_c = tl.where(iw_num >= 0, iw_num, 0)
            vw = (iw_num >= 0) & ((iw_c % STRIDE) == 0)
            iw = iw_c // STRIDE
            vw = vw & (iw < W)
            valid = vh & vw & pmask
            sp_off = x_base + (ih * W + iw).to(tl.int64)  # [BLOCK_P]
            tap = kh * KW + kw
            for ci0 in range(0, C_IN, BLOCK_K):
                ci = ci0 + offs_k
                ci_mask = ci < C_IN
                x_ptrs = x_ptr + sp_off[:, None] + (ci.to(tl.int64) * H * W)[None, :]
                xt = tl.load(x_ptrs, mask=valid[:, None] & ci_mask[None, :], other=0.0)
                w_rows = (tap * C_IN + ci).to(tl.int64)
                w_ptrs = w_ptr + w_rows[:, None] * C_OUT + offs_co[None, :]
                wt = tl.load(w_ptrs, mask=ci_mask[:, None] & co_mask[None, :], other=0.0)
                acc += tl.dot(xt, wt)

    cb = tl.load(cb_ptr + offs_co, mask=co_mask, other=0.0).to(tl.float32)
    acc = acc + cb[None, :]
    acc = tl.where(co_mask[None, :], acc, float("-inf"))
    m = tl.max(acc, axis=1)
    e = tl.exp(acc - m[:, None])
    e = tl.where(co_mask[None, :], e, 0.0)
    s = tl.sum(e, axis=1)
    sm = e / s[:, None]

    b = tl.load(b_ptr + offs_co, mask=co_mask, other=0.0).to(tl.float32)
    y = (sm + b[None, :]) * scale
    y = 1.0 / (1.0 + tl.exp(-y))

    out_off = (n64 * C_OUT * ohw + rem.to(tl.int64))[:, None] + (offs_co.to(tl.int64) * ohw)[None, :]
    tl.store(out_ptr + out_off, y.to(out_ptr.dtype.element_ty),
             mask=pmask[:, None] & co_mask[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        conv = self.conv_transpose
        x = x.contiguous()
        N, C_IN, H, W = x.shape
        w = conv.weight  # [C_IN, C_OUT, KH, KW]
        C_OUT = w.shape[1]
        KH, KW = conv.kernel_size
        stride = conv.stride[0]
        pad = conv.padding[0]
        opad = conv.output_padding[0]
        OH = (H - 1) * stride - 2 * pad + KH + opad
        OW = (W - 1) * stride - 2 * pad + KW + opad

        # weight reorganized to [KH, KW, C_IN, C_OUT] -> rows (kh*KW+kw)*C_IN+ci
        w_r = w.to(x.dtype).permute(2, 3, 0, 1).contiguous().reshape(KH * KW * C_IN, C_OUT)
        if conv.bias is not None:
            cb = conv.bias.contiguous()
        else:
            cb = torch.zeros(C_OUT, device=x.device, dtype=torch.float32)
        b = self.bias.reshape(-1).contiguous()

        out = torch.empty((N, C_OUT, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_P = 64
        BLOCK_K = 32
        BLOCK_CO = max(16, triton.next_power_of_2(C_OUT))
        grid = (triton.cdiv(N * OH * OW, BLOCK_P),)
        convt_softmax_bias_sigmoid_kernel[grid](
            x, w_r, cb, b, out,
            N, C_IN, H, W, C_OUT, OH, OW,
            float(self.scaling_factor),
            KH=KH, KW=KW, STRIDE=stride, PAD=pad,
            BLOCK_P=BLOCK_P, BLOCK_K=BLOCK_K, BLOCK_CO=BLOCK_CO,
            num_warps=4,
        )
        return out
