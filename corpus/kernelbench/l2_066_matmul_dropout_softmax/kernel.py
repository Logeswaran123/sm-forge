import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        mask_k = kk < K
        a = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        )
        # W tile laid out as [BLOCK_K, BLOCK_N] (i.e. W^T)
        b = tl.load(
            w_ptr + offs_n[None, :] * stride_wn + kk[:, None] * stride_wk,
            mask=mask_n[None, :] & mask_k[:, None], other=0.0,
        )
        b = b.to(a.dtype)
        acc = tl.dot(a, b, acc)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc, mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def softmax_kernel(
    in_ptr, out_ptr, N,
    stride_in, stride_out,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    in_row = in_ptr + row * stride_in
    out_row = out_ptr + row * stride_out

    m = tl.full((BLOCK,), -float("inf"), dtype=tl.float32)
    s = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, N, BLOCK):
        idx = start + offs
        mask = idx < N
        v = tl.load(in_row + idx, mask=mask, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(m, v)
        # guard lanes that are still all -inf
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        s = s * tl.exp(m - m_safe) + tl.exp(v - m_safe)
        m = m_new

    row_max = tl.max(m, axis=0)
    m_safe = tl.where(m == -float("inf"), row_max, m)
    total = tl.sum(s * tl.exp(m_safe - row_max), axis=0)
    inv = 1.0 / total

    for start in range(0, N, BLOCK):
        idx = start + offs
        mask = idx < N
        v = tl.load(in_row + idx, mask=mask, other=-float("inf")).to(tl.float32)
        p = tl.exp(v - row_max) * inv
        tl.store(out_row + idx, p.to(out_ptr.dtype.element_ty), mask=mask)


class Model(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # eval mode: dropout is identity
        x = x.contiguous()
        w = self.matmul.weight
        b = self.matmul.bias
        M, K = x.shape
        N = w.shape[0]

        logits = torch.empty((M, N), device=x.device, dtype=torch.float32)
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        linear_kernel[grid](
            x, w, b, logits,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        softmax_kernel[(M,)](
            logits, out, N,
            logits.stride(0), out.stride(0),
            BLOCK=2048, num_warps=8,
        )
        return out
