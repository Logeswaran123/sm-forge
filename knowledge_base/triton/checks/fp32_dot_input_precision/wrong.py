import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------- Large-batch path: emulate fp32 GEMM (sequential-k FMA) ----------------
@triton.jit
def _gemm_rowsum_kernel(x_ptr, w_ptr, part_ptr, B, H, K,
                        stride_xb, stride_xk, stride_wh, stride_wk,
                        num_n_blocks,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < B
    mask_n = offs_n < H
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a = tl.load(x_ptr + offs_m[:, None] * stride_xb + offs_k[None, :] * stride_xk,
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(w_ptr + offs_n[None, :] * stride_wh + offs_k[:, None] * stride_wk,
                    mask=mask_n[None, :] & mask_k[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
    y = acc * 0.5  # exact division by 2
    s = tl.sum(y.to(tl.float64), axis=1)
    tl.store(part_ptr + offs_m * num_n_blocks + pid_n, s, mask=mask_m)


@triton.jit
def _finalize_kernel(part_ptr, out_ptr, num_n_blocks, scale,
                     BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float64)
    for i0 in range(0, num_n_blocks, BLOCK):
        offs = i0 + tl.arange(0, BLOCK)
        mask = offs < num_n_blocks
        acc += tl.load(part_ptr + pid * num_n_blocks + offs, mask=mask, other=0.0)
    total = tl.sum(acc, axis=0)
    res = total.to(tl.float32) * scale
    tl.store(out_ptr + pid, res)


# ---------------- Small-batch path: high-precision column-sum trick ----------------
@triton.jit
def _colsum_kernel(w_ptr, out_ptr, H, K, stride_wh, stride_wk,
                   BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_k = pid * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    acc = tl.zeros((BLOCK_H, BLOCK_K), dtype=tl.float64)
    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        ptrs = w_ptr + offs_h[:, None] * stride_wh + offs_k[None, :] * stride_wk
        w = tl.load(ptrs, mask=mask_h[:, None] & mask_k[None, :], other=0.0)
        acc += w.to(tl.float64)
    s = tl.sum(acc, axis=0)
    tl.store(out_ptr + offs_k, s, mask=mask_k)


@triton.jit
def _rowdot_kernel(x_ptr, v_ptr, out_ptr, K, stride_xb, stride_xk, scale,
                   BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float64)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        xv = tl.load(x_ptr + pid * stride_xb + offs_k * stride_xk, mask=mask_k, other=0.0)
        vv = tl.load(v_ptr + offs_k, mask=mask_k, other=0.0)
        acc += xv.to(tl.float64) * vv
    total = tl.sum(acc, axis=0) * 0.5
    res = total.to(tl.float32) * scale
    tl.store(out_ptr + pid, res)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        w = self.weight
        H, K = w.shape
        B = x.shape[0]
        out = torch.empty((B, 1), device=x.device, dtype=torch.float32)
        scale = float(self.scaling_factor)

        if B <= 64:
            colsum = torch.empty((K,), device=x.device, dtype=torch.float64)
            BLOCK_K1, BLOCK_H = 64, 32
            _colsum_kernel[(triton.cdiv(K, BLOCK_K1),)](
                w, colsum, H, K, w.stride(0), w.stride(1),
                BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K1,
            )
            _rowdot_kernel[(B,)](
                x, colsum, out, K, x.stride(0), x.stride(1), scale,
                BLOCK_K=512,
            )
        else:
            BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            num_n_blocks = triton.cdiv(H, BLOCK_N)
            partials = torch.empty((B, num_n_blocks), device=x.device, dtype=torch.float64)
            grid = (triton.cdiv(B, BLOCK_M), num_n_blocks)
            _gemm_rowsum_kernel[grid](
                x, w, partials, B, H, K,
                x.stride(0), x.stride(1), w.stride(0), w.stride(1),
                num_n_blocks,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )
            _finalize_kernel[(B,)](
                partials, out, num_n_blocks, scale,
                BLOCK=128,
            )

        if out.dtype != x.dtype:
            out = out.to(x.dtype)
        return out
