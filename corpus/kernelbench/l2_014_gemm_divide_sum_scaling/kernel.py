import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _colsum_kernel(w_ptr, out_ptr, H, K, stride_wh, stride_wk,
                   BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr):
    # out[k] = sum_j W[j, k]
    pid = tl.program_id(0)
    offs_k = pid * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    acc = tl.zeros([BLOCK_H, BLOCK_K], dtype=tl.float32)
    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        ptrs = w_ptr + offs_h[:, None] * stride_wh + offs_k[None, :] * stride_wk
        w = tl.load(ptrs, mask=mask_h[:, None] & mask_k[None, :], other=0.0)
        acc += w.to(tl.float32)
    s = tl.sum(acc, axis=0)
    tl.store(out_ptr + offs_k, s, mask=mask_k)


@triton.jit
def _rowdot_kernel(x_ptr, c_ptr, out_ptr, K, stride_xb, stride_xk, scale,
                   BLOCK_K: tl.constexpr):
    # out[b] = scale * sum_k x[b, k] * c[k]
    b = tl.program_id(0)
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        xv = tl.load(x_ptr + b * stride_xb + offs_k * stride_xk, mask=mask_k, other=0.0)
        cv = tl.load(c_ptr + offs_k, mask=mask_k, other=0.0)
        acc += xv.to(tl.float32) * cv
    total = tl.sum(acc, axis=0)
    tl.store(out_ptr + b, total * scale)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        w = self.weight
        H, K = w.shape
        B = x.shape[0]

        colsum = torch.empty((K,), device=x.device, dtype=torch.float32)
        BLOCK_H = 32
        BLOCK_K = 64
        grid = (triton.cdiv(K, BLOCK_K),)
        _colsum_kernel[grid](w, colsum, H, K, w.stride(0), w.stride(1),
                             BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K)

        out = torch.empty((B, 1), device=x.device, dtype=torch.float32)
        scale = float(self.scaling_factor) * 0.5
        _rowdot_kernel[(B,)](x, colsum, out, K, x.stride(0), x.stride(1), scale,
                             BLOCK_K=1024)
        return out.to(x.dtype)
