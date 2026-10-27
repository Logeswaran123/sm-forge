import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_linear_act_kernel(
    x_ptr, w_ptr, bias_ptr, add_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        mask_k = kk < K
        a_ptrs = x_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk
        b_ptrs = w_ptr + offs_n[None, :] * stride_wn + kk[:, None] * stride_wk
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        b = b.to(a.dtype)
        acc = tl.dot(a, b, acc)

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    addv = tl.load(add_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    y = acc + bias[None, :] + addv[None, :]

    # Swish
    sig = 1.0 / (1.0 + tl.exp(-y))
    y = y * sig
    # Tanh (stable form)
    y = 1.0 - 2.0 / (tl.exp(2.0 * y) + 1.0)
    # GELU (exact, erf)
    y = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    # Hardtanh
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)

    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(o_ptrs, y.to(out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


class Model(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))

    def forward(self, x):
        in_features = self.matmul.in_features
        out_features = self.matmul.out_features
        orig_shape = x.shape
        x2 = x.reshape(-1, in_features)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        M, K = x2.shape
        N = out_features
        w = self.matmul.weight
        b = self.matmul.bias
        addv = self.add_value.reshape(-1)
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        _fused_linear_act_kernel[grid](
            x2, w, b, addv, out,
            M, N, K,
            x2.stride(0), x2.stride(1),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_M=8,
            num_warps=8, num_stages=3,
        )
        return out.reshape(*orig_shape[:-1], N)
