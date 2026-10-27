import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_kernel(
    a_ptr, w_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    APPLY_SIGMOID: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # W is [N, K]; load a [BLOCK_K, BLOCK_N] tile of W^T
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + offs_k) < K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, w, acc)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    if APPLY_SIGMOID:
        acc = 1.0 / (1.0 + tl.exp(-acc))

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


@triton.jit
def logsumexp_kernel(
    x_ptr, out_ptr, N, stride_xm,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=float("-inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.where(mask, tl.exp(x - m), 0.0)
    s = tl.sum(e, axis=0)
    res = m + tl.log(s)
    tl.store(out_ptr + row, res.to(out_ptr.dtype.element_ty))


def _linear(a, weight, bias, out, apply_sigmoid):
    M, K = a.shape
    N = weight.shape[0]
    BM, BN, BK = 128, 128, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    linear_kernel[grid](
        a, weight, bias, out,
        M, N, K,
        a.stride(0), a.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        APPLY_SIGMOID=apply_sigmoid,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        num_warps=8, num_stages=3,
    )
    return out


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.contiguous()
        M = x.shape[0]
        w1 = self.linear1.weight
        b1 = self.linear1.bias
        w2 = self.linear2.weight
        b2 = self.linear2.bias
        H = w1.shape[0]
        O = w2.shape[0]

        h = torch.empty((M, H), device=x.device, dtype=x.dtype)
        _linear(x, w1, b1, h, True)

        logits = torch.empty((M, O), device=x.device, dtype=torch.float32)
        _linear(h, w2, b2, logits, False)

        out = torch.empty((M,), device=x.device, dtype=x.dtype)
        BLOCK = triton.next_power_of_2(O)
        logsumexp_kernel[(M,)](logits, out, O, logits.stride(0), BLOCK_N=BLOCK, num_warps=4)
        return out
