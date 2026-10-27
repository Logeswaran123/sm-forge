import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _spatial_sum_kernel(x_ptr, s_ptr, HW, BLOCK: tl.constexpr):
    # one program per (b, ci) row of length HW
    row = tl.program_id(0)
    base = x_ptr + row.to(tl.int64) * HW
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, HW, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < HW
        v = tl.load(base + offs, mask=m, other=0.0)
        acc += v.to(tl.float32)
    tl.store(s_ptr + row, tl.sum(acc, axis=0))


@triton.jit
def _epilogue_kernel(s_ptr, w_ptr, cb_ptr, b_ptr, out_ptr,
                     CIN, COUT, KK, inv_area,
                     HAS_CB: tl.constexpr,
                     BLOCK_CI: tl.constexpr, BLOCK_O: tl.constexpr):
    b = tl.program_id(0)
    ci = tl.arange(0, BLOCK_CI)
    o = tl.arange(0, BLOCK_O)
    ci_m = ci < CIN
    o_m = o < COUT

    s = tl.load(s_ptr + b * CIN + ci, mask=ci_m, other=0.0)  # [BLOCK_CI] fp32

    # weight layout: [CIN, COUT, K, K] contiguous
    wmask = ci_m[:, None] & o_m[None, :]
    wbase = w_ptr + (ci[:, None] * COUT + o[None, :]) * KK
    wsum = tl.zeros([BLOCK_CI, BLOCK_O], dtype=tl.float32)
    for k in range(0, KK):
        wv = tl.load(wbase + k, mask=wmask, other=0.0)
        wsum += wv.to(tl.float32)

    val = tl.sum(s[:, None] * wsum, axis=0) * inv_area  # [BLOCK_O]
    if HAS_CB:
        cb = tl.load(cb_ptr + o, mask=o_m, other=0.0).to(tl.float32)
        val += cb
    bb = tl.load(b_ptr + o, mask=o_m, other=0.0).to(tl.float32)
    val += bb

    val = tl.where(o_m, val, float("-inf"))
    mx = tl.max(val, axis=0)
    e = tl.where(o_m, tl.exp(val - mx), 0.0)
    lse = mx + tl.log(tl.sum(e, axis=0))
    res = lse * 10.0
    tl.store(out_ptr + b, res.to(out_ptr.dtype.element_ty))


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous()
        B, CIN, H, W = x.shape
        w = self.conv_transpose.weight.contiguous()  # [CIN, COUT, K, K]
        COUT = w.shape[1]
        KH, KW = w.shape[2], w.shape[3]
        HOUT = H + KH - 1
        WOUT = W + KW - 1
        HW = H * W

        s = torch.empty((B * CIN,), device=x.device, dtype=torch.float32)
        _spatial_sum_kernel[(B * CIN,)](x, s, HW, BLOCK=4096, num_warps=8)

        out = torch.empty((B, 1), device=x.device, dtype=x.dtype)
        cb = self.conv_transpose.bias
        has_cb = cb is not None
        cb_arg = cb.contiguous() if has_cb else w
        bias = self.bias.contiguous().view(-1)

        BLOCK_CI = max(triton.next_power_of_2(CIN), 2)
        BLOCK_O = max(triton.next_power_of_2(COUT), 2)
        _epilogue_kernel[(B,)](
            s, w, cb_arg, bias, out,
            CIN, COUT, KH * KW, 1.0 / float(HOUT * WOUT),
            HAS_CB=has_cb, BLOCK_CI=BLOCK_CI, BLOCK_O=BLOCK_O, num_warps=4,
        )
        return out
