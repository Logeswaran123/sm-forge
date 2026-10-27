import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def convt_min_kernel(
    x_ptr, w_ptr, cb_ptr, mid_ptr,
    Cin, Cout, H, W, Hout, Wout,
    sxn, sxc, sxh, sxw,
    swi, swo, swh, sww,
    HAS_CB: tl.constexpr,
    S: tl.constexpr, P: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_W: tl.constexpr, BLOCK_CI: tl.constexpr, BLOCK_CO: tl.constexpr,
):
    pid_w = tl.program_id(0)
    oh = tl.program_id(1)
    n = tl.program_id(2)

    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    ow_ok = ow < Wout
    co = tl.arange(0, BLOCK_CO)
    co_ok = co < Cout
    ci_r = tl.arange(0, BLOCK_CI)

    acc = tl.zeros((BLOCK_W, BLOCK_CO), dtype=tl.float32)
    x_base = x_ptr + n * sxn

    for kh in tl.static_range(KH):
        th = oh + P - kh
        th_c = tl.maximum(th, 0)
        ih = th_c // S
        h_ok = (th >= 0) & ((th_c % S) == 0) & (ih < H)
        for kw in tl.static_range(KW):
            tw = ow + P - kw
            tw_c = tl.maximum(tw, 0)
            iw = tw_c // S
            w_ok = (tw >= 0) & ((tw_c % S) == 0) & (iw < W) & ow_ok & h_ok
            for ci0 in range(0, Cin, BLOCK_CI):
                ci = ci0 + ci_r
                ci_ok = ci < Cin
                xp = x_base + ci[None, :] * sxc + ih * sxh + iw[:, None] * sxw
                xv = tl.load(xp, mask=w_ok[:, None] & ci_ok[None, :], other=0.0)
                wp = w_ptr + ci[:, None] * swi + co[None, :] * swo + kh * swh + kw * sww
                wv = tl.load(wp, mask=ci_ok[:, None] & co_ok[None, :], other=0.0)
                acc = tl.dot(xv.to(tl.float16), wv.to(tl.float16), acc)

    if HAS_CB:
        cb = tl.load(cb_ptr + co, mask=co_ok, other=0.0).to(tl.float32)
        acc = acc + cb[None, :]
    acc = tl.where(co_ok[None, :], acc, float("inf"))
    m = tl.min(acc, axis=1)
    tl.store(mid_ptr + (n * Hout + oh) * Wout + ow, m, mask=ow_ok)


@triton.jit
def sum_gelu_bias_kernel(
    mid_ptr, b_ptr, out_ptr, Hout, Wout,
    BLOCK: tl.constexpr,
):
    n = tl.program_id(0)
    pid = tl.program_id(1)
    ow = pid * BLOCK + tl.arange(0, BLOCK)
    ok = ow < Wout
    s = tl.zeros((BLOCK,), dtype=tl.float32)
    base = mid_ptr + n * Hout * Wout
    for h in range(0, Hout):
        s += tl.load(base + h * Wout + ow, mask=ok, other=0.0)
    g = 0.5 * s * (1.0 + tl.math.erf(s * 0.7071067811865476))
    b = tl.load(b_ptr).to(tl.float32)
    y = g + b
    tl.store(out_ptr + n * Wout + ow, y.to(out_ptr.dtype.element_ty), mask=ok)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        conv = self.conv_transpose
        x = x.contiguous()
        w = conv.weight.contiguous()
        N, Cin, H, W = x.shape
        Cout = w.shape[1]
        KH, KW = conv.kernel_size
        S = conv.stride[0]
        P = conv.padding[0]
        OP = conv.output_padding[0]
        Hout = (H - 1) * S - 2 * P + KH + OP
        Wout = (W - 1) * S - 2 * P + KW + OP

        mid = torch.empty((N, Hout, Wout), device=x.device, dtype=torch.float32)
        has_cb = conv.bias is not None
        cb = conv.bias if has_cb else w

        BLOCK_W = 64
        BLOCK_CI = 32
        BLOCK_CO = max(16, triton.next_power_of_2(Cout))
        grid = (triton.cdiv(Wout, BLOCK_W), Hout, N)
        convt_min_kernel[grid](
            x, w, cb, mid,
            Cin, Cout, H, W, Hout, Wout,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
            HAS_CB=has_cb, S=S, P=P, KH=KH, KW=KW,
            BLOCK_W=BLOCK_W, BLOCK_CI=BLOCK_CI, BLOCK_CO=BLOCK_CO,
            num_warps=4,
        )

        out = torch.empty((N, 1, 1, Wout), device=x.device, dtype=x.dtype)
        BLOCK = 128
        grid2 = (N, triton.cdiv(Wout, BLOCK))
        sum_gelu_bias_kernel[grid2](mid, self.bias, out, Hout, Wout, BLOCK=BLOCK, num_warps=4)
        return out
