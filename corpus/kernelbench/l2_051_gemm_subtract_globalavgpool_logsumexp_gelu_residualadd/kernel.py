import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _col_mean_kernel(w_ptr, out_ptr, N, K, stride_wn, stride_wk,
                     BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_k = pid * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        tile = tl.load(ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(tile, axis=0)
    tl.store(out_ptr + offs_k, acc / N, mask=mask_k)


@triton.jit
def _const_kernel(b_ptr, s_ptr, out_ptr, N, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for n0 in range(0, N, BLOCK):
        offs = n0 + tl.arange(0, BLOCK)
        m = offs < N
        s = tl.load(s_ptr + offs, mask=m, other=0.0).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(b_ptr + offs, mask=m, other=0.0).to(tl.float32)
            acc += b - s
        else:
            acc -= s
    total = tl.sum(acc, axis=0)
    tl.store(out_ptr, total / N)


@triton.jit
def _row_kernel(x_ptr, wm_ptr, c_ptr, out_ptr, K, stride_xm, stride_om,
                BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        m = offs < K
        xv = tl.load(x_ptr + row * stride_xm + offs, mask=m, other=0.0).to(tl.float32)
        wv = tl.load(wm_ptr + offs, mask=m, other=0.0)
        acc += xv * wv
    v = tl.sum(acc, axis=0) + tl.load(c_ptr)
    # logsumexp over a size-1 dim is identity; exact GELU
    g = 0.5 * v * (1.0 + tl.math.erf(v * 0.7071067811865476))
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        m = offs < K
        xv = tl.load(x_ptr + row * stride_xm + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * stride_om + offs,
                 (xv + g).to(out_ptr.dtype.element_ty), mask=m)


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        x = x.contiguous()
        B, K = x.shape
        W = self.gemm.weight
        N = W.shape[0]
        dev = x.device

        wmean = torch.empty(K, device=dev, dtype=torch.float32)
        BLOCK_KW = 64
        _col_mean_kernel[(triton.cdiv(K, BLOCK_KW),)](
            W, wmean, N, K, W.stride(0), W.stride(1),
            BLOCK_N=64, BLOCK_K=BLOCK_KW)

        c = torch.empty(1, device=dev, dtype=torch.float32)
        has_bias = self.gemm.bias is not None
        b = self.gemm.bias if has_bias else self.subtract
        _const_kernel[(1,)](b, self.subtract, c, N, HAS_BIAS=has_bias, BLOCK=1024)

        out = torch.empty_like(x)
        _row_kernel[(B,)](x, wmean, c, out, K, x.stride(0), out.stride(0),
                          BLOCK_K=1024)
        return out
