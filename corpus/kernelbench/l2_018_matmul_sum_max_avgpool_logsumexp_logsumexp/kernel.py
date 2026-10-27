import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def colsum_kernel(w_ptr, out_ptr, N, K, stride_wn, stride_wk,
                  BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_k = pid * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask = (offs_n[:, None] < N) & mask_k[None, :]
        ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        tile = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        acc += tile
    s = tl.sum(acc, axis=0)
    tl.store(out_ptr + offs_k, s, mask=mask_k)


@triton.jit
def row_dot_kernel(x_ptr, wsum_ptr, b_ptr, out_ptr, K, N, stride_xm, stride_xk,
                   BLOCK_K: tl.constexpr, BLOCK_B: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        m = offs < K
        xv = tl.load(x_ptr + pid * stride_xm + offs * stride_xk, mask=m, other=0.0).to(tl.float32)
        wv = tl.load(wsum_ptr + offs, mask=m, other=0.0)
        acc += xv * wv
    bacc = tl.zeros((BLOCK_B,), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_B):
        offs = n0 + tl.arange(0, BLOCK_B)
        m = offs < N
        bacc += tl.load(b_ptr + offs, mask=m, other=0.0).to(tl.float32)
    res = tl.sum(acc, axis=0) + tl.sum(bacc, axis=0)
    tl.store(out_ptr + pid, res)


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        w = self.linear.weight  # (N, K)
        b = self.linear.bias.contiguous()
        N, K = w.shape
        M = x.shape[0]

        wsum = torch.empty((K,), device=x.device, dtype=torch.float32)
        BLOCK_K_C = 64
        BLOCK_N_C = 64
        colsum_kernel[(triton.cdiv(K, BLOCK_K_C),)](
            w, wsum, N, K, w.stride(0), w.stride(1),
            BLOCK_N=BLOCK_N_C, BLOCK_K=BLOCK_K_C, num_warps=4)

        out = torch.empty((M, 1), device=x.device, dtype=torch.float32)
        row_dot_kernel[(M,)](
            x, wsum, b, out, K, N, x.stride(0), x.stride(1),
            BLOCK_K=1024, BLOCK_B=1024, num_warps=4)
        return out.to(x.dtype)
