import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mish(x):
    # softplus with PyTorch threshold=20
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
    sp = tl.where(x > 20.0, x, sp)
    # tanh(sp) for sp >= 0, stable form
    e = tl.exp(-2.0 * sp)
    t = (1.0 - e) / (1.0 + e)
    return x * t


@triton.jit
def linear_mish_mish_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    HAS_BIAS: tl.constexpr,
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

        # W tile laid out as [BLOCK_K, BLOCK_N] (i.e. W^T)
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + kk[:, None] * stride_wk
        w_mask = (offs_n[None, :] < N) & (kk[:, None] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        a = a.to(tl.float16)
        w = w.to(tl.float16)
        acc += tl.dot(a, w)

    if HAS_BIAS:
        b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += b[None, :]

    acc = _mish(acc)
    acc = _mish(acc)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        w = self.linear.weight
        b = self.linear.bias
        K = x.shape[-1]
        orig_shape = x.shape
        x2 = x.reshape(-1, K)
        M = x2.shape[0]
        N = w.shape[0]

        y = torch.empty((M, N), device=x.device, dtype=x.dtype)

        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        has_bias = b is not None
        linear_mish_mish_kernel[grid](
            x2, w, b if has_bias else w, y,
            M, N, K,
            x2.stride(0), x2.stride(1),
            w.stride(0), w.stride(1),
            y.stride(0), y.stride(1),
            HAS_BIAS=has_bias,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )
        return y.reshape(*orig_shape[:-1], N)
