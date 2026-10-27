import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gemm_mul_lrelu_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    multiplier, negative_slope,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        a_ptrs = x_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk
        a_mask = (offs_m[:, None] < M) & (kk[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        # B tile: [BLOCK_K, BLOCK_N] view of W^T (W is [N, K])
        b_ptrs = w_ptr + offs_n[None, :] * stride_wn + kk[:, None] * stride_wk
        b_mask = (offs_n[None, :] < N) & (kk[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b.to(a.dtype), acc)

    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    acc = acc * multiplier
    acc = tl.where(acc >= 0, acc, acc * negative_slope)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


class Model(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        w = self.gemm.weight
        b = self.gemm.bias
        K = x.shape[-1]
        lead = x.shape[:-1]
        x2 = x.reshape(-1, K)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        if w.dtype != x2.dtype:
            w = w.to(x2.dtype)
        M = x2.shape[0]
        N = w.shape[0]
        y = torch.empty((M, N), device=x.device, dtype=x.dtype)

        BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 64
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        _gemm_mul_lrelu_kernel[grid](
            x2, w, b, y,
            M, N, K,
            x2.stride(0), x2.stride(1),
            w.stride(0), w.stride(1),
            y.stride(0), y.stride(1),
            float(self.multiplier), float(self.leaky_relu.negative_slope),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_M=8,
            num_warps=4, num_stages=3,
        )
        return y.view(*lead, N)
