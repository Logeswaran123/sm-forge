import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_gelu_kernel(
    x_ptr, w_ptr, b_ptr, h_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_hm, stride_hn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        a_ptrs = x_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk
        a_mask = (offs_m[:, None] < M) & (kk[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_ptrs = w_ptr + offs_n[None, :] * stride_wn + kk[:, None] * stride_wk
        b_mask = (offs_n[None, :] < N) & (kk[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b.to(a.dtype))

    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    # exact GELU
    acc = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))

    h_ptrs = h_ptr + offs_m[:, None] * stride_hm + offs_n[None, :] * stride_hn
    h_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(h_ptrs, acc, mask=h_mask)


@triton.jit
def softmax_kernel(
    h_ptr, out_ptr, N,
    stride_hm, stride_om,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    h_row = h_ptr + row * stride_hm
    o_row = out_ptr + row * stride_om

    m_vec = tl.full((BLOCK,), float("-inf"), tl.float32)
    s_vec = tl.zeros((BLOCK,), tl.float32)
    for n0 in range(0, N, BLOCK):
        idx = n0 + offs
        v = tl.load(h_row + idx, mask=idx < N, other=float("-inf"))
        m_new = tl.maximum(m_vec, v)
        # guard against -inf - -inf
        scale = tl.where(m_new == float("-inf"), 0.0, tl.exp(m_vec - m_new))
        e = tl.where(m_new == float("-inf"), 0.0, tl.exp(v - m_new))
        s_vec = s_vec * scale + e
        m_vec = m_new

    m = tl.max(m_vec, axis=0)
    s = tl.sum(tl.where(m_vec == float("-inf"), 0.0, s_vec * tl.exp(m_vec - m)), axis=0)
    inv = 1.0 / s

    for n0 in range(0, N, BLOCK):
        idx = n0 + offs
        mask = idx < N
        v = tl.load(h_row + idx, mask=mask, other=float("-inf"))
        p = tl.exp(v - m) * inv
        tl.store(o_row + idx, p.to(out_ptr.dtype.element_ty), mask=mask)


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.contiguous()
        w = self.linear.weight
        b = self.linear.bias
        M, K = x.shape
        N = w.shape[0]

        h = torch.empty((M, N), device=x.device, dtype=torch.float32)
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        linear_gelu_kernel[grid](
            x, w, b, h,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            h.stride(0), h.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        softmax_kernel[(M,)](
            h, out, N,
            h.stride(0), out.stride(0),
            BLOCK=1024,
            num_warps=4,
        )
        return out
