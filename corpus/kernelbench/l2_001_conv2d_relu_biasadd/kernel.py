import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_relu_bias_kernel(
    x_ptr, w_ptr, cb_ptr, b_ptr, out_ptr,
    NB, C, H, W, OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    HAS_CONV_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    OHW = OH * OW
    M = NB * OHW
    K = C * KH * KW
    HW = H * W

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < OC

    n_img = offs_m // OHW
    rem = offs_m % OHW
    oh = rem // OW
    ow = rem % OW

    x_base = n_img.to(tl.int64) * C * HW + (oh * W + ow).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        c = offs_k // (KH * KW)
        r = offs_k % (KH * KW)
        kh = r // KW
        kw = r % KW
        k_off = (c * HW + kh * W + kw).to(tl.int64)

        a_ptrs = x_ptr + x_base[:, None] + k_off[None, :]
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        w_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]
        wt = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        wt = wt.to(a.dtype)

        acc = tl.dot(a, wt, acc)

    if HAS_CONV_BIAS:
        cb = tl.load(cb_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc = acc + cb[None, :]
    acc = tl.maximum(acc, 0.0)
    b = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc + b[None, :]

    out_off = (n_img.to(tl.int64) * OC * OHW + rem.to(tl.int64))[:, None] \
        + (offs_n.to(tl.int64) * OHW)[None, :]
    tl.store(out_ptr + out_off, acc.to(out_ptr.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        NB, C, H, W = x.shape
        OC, _, KH, KW = w.shape
        OH = H - KH + 1
        OW = W - KW + 1
        out = torch.empty((NB, OC, OH, OW), device=x.device, dtype=x.dtype)

        has_cb = self.conv.bias is not None
        cb = self.conv.bias.contiguous() if has_cb else self.bias.reshape(-1).contiguous()
        b = self.bias.reshape(-1).contiguous()

        BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 32
        M = NB * OH * OW
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(OC, BLOCK_N))
        conv_relu_bias_kernel[grid](
            x, w, cb, b, out,
            NB, C, H, W, OC, OH, OW,
            KH=KH, KW=KW, HAS_CONV_BIAS=has_cb,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )
        return out
