import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _spatial_sum_kernel(x_ptr, s_ptr, L, BLOCK: tl.constexpr):
    # one program per (n, ci) row of length L
    row = tl.program_id(0)
    base = x_ptr + row.to(tl.int64) * L
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < L
        v = tl.load(base + offs, mask=m, other=0.0).to(tl.float32)
        acc += v
    tl.store(s_ptr + row, tl.sum(acc, axis=0))


@triton.jit
def _weight_sum_kernel(w_ptr, ws_ptr, KV, BLOCK: tl.constexpr):
    # one program per (ci, co) pair; sums the KV kernel taps
    row = tl.program_id(0)
    base = w_ptr + row.to(tl.int64) * KV
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, KV, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < KV
        v = tl.load(base + offs, mask=m, other=0.0).to(tl.float32)
        acc += v
    tl.store(ws_ptr + row, tl.sum(acc, axis=0))


@triton.jit
def _final_kernel(s_ptr, ws_ptr, bias_ptr, rm_ptr, rv_ptr, g_ptr, b_ptr, out_ptr,
                  CIN, COUT, inv_nout, scale, eps,
                  BLOCK_CI: tl.constexpr, BLOCK_CO: tl.constexpr):
    n = tl.program_id(0)
    pco = tl.program_id(1)
    offs_co = pco * BLOCK_CO + tl.arange(0, BLOCK_CO)
    mco = offs_co < COUT
    acc = tl.zeros([BLOCK_CI, BLOCK_CO], dtype=tl.float32)
    for start in range(0, CIN, BLOCK_CI):
        offs_ci = start + tl.arange(0, BLOCK_CI)
        mci = offs_ci < CIN
        s = tl.load(s_ptr + n * CIN + offs_ci, mask=mci, other=0.0)
        w = tl.load(ws_ptr + offs_ci[:, None] * COUT + offs_co[None, :],
                    mask=mci[:, None] & mco[None, :], other=0.0)
        acc += s[:, None] * w
    tot = tl.sum(acc, axis=0)
    bias = tl.load(bias_ptr + offs_co, mask=mco, other=0.0).to(tl.float32)
    mean_y = tot * inv_nout + bias
    rm = tl.load(rm_ptr + offs_co, mask=mco, other=0.0).to(tl.float32)
    rv = tl.load(rv_ptr + offs_co, mask=mco, other=1.0).to(tl.float32)
    g = tl.load(g_ptr + offs_co, mask=mco, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offs_co, mask=mco, other=0.0).to(tl.float32)
    y = (mean_y * scale - rm) * tl.rsqrt(rv + eps) * g + b
    tl.store(out_ptr + n * COUT + offs_co, y.to(out_ptr.dtype.element_ty), mask=mco)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = x.contiguous()
        N, CIN, D, H, W = x.shape
        w = self.conv_transpose.weight.contiguous()  # [CIN, COUT, kd, kh, kw]
        COUT = w.shape[1]
        kd, kh, kw = w.shape[2], w.shape[3], w.shape[4]
        KV = kd * kh * kw
        Dout, Hout, Wout = D + kd - 1, H + kh - 1, W + kw - 1
        nout = Dout * Hout * Wout
        L = D * H * W
        dev = x.device

        s = torch.empty((N * CIN,), device=dev, dtype=torch.float32)
        _spatial_sum_kernel[(N * CIN,)](x, s, L, BLOCK=1024, num_warps=4)

        ws = torch.empty((CIN * COUT,), device=dev, dtype=torch.float32)
        _weight_sum_kernel[(CIN * COUT,)](w, ws, KV, BLOCK=128, num_warps=1)

        bn = self.batch_norm
        bias = self.conv_transpose.bias
        out = torch.empty((N, COUT, 1, 1, 1), device=dev, dtype=x.dtype)
        BLOCK_CO = 128
        grid = (N, triton.cdiv(COUT, BLOCK_CO))
        _final_kernel[grid](s, ws, bias.contiguous(), bn.running_mean.contiguous(),
                            bn.running_var.contiguous(), bn.weight.contiguous(),
                            bn.bias.contiguous(), out,
                            CIN, COUT, 1.0 / float(nout), float(self.scale_factor), float(bn.eps),
                            BLOCK_CI=32, BLOCK_CO=BLOCK_CO, num_warps=4)
        return out
