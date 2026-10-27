import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_act_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    HAS_BIAS: tl.constexpr,
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

    mask_m = offs_m < M
    mask_n = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # W is [N, K]; load a [BLOCK_K, BLOCK_N] tile (i.e. W^T)
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + offs_k) < K
        a = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(w_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    y = acc
    # swish
    y = y * (1.0 / (1.0 + tl.exp(-y)))
    # divide
    y = y / 2.0
    # clamp
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)
    # tanh (input in [-1, 1], numerically safe)
    y = 1.0 - 2.0 / (tl.exp(2.0 * y) + 1.0)
    # clamp
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, y.to(out_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        w = self.gemm.weight
        b = self.gemm.bias
        x = x.contiguous()
        M, K = x.shape
        N = w.shape[0]
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        has_bias = b is not None
        _linear_act_kernel[grid](
            x, w, b if has_bias else w, out,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1),
            HAS_BIAS=has_bias,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_M=8,
            num_warps=8, num_stages=3,
        )
        return out
