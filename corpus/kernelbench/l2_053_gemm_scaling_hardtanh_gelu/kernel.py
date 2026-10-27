import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_scale_hardtanh_gelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    scaling, ht_min, ht_max,
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

    m_mask = offs_m < M
    n_mask = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # W is [N, K]; load a transposed tile [BLOCK_K, BLOCK_N]
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + offs_k) < K
        a = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        b = b.to(a.dtype)
        acc = tl.dot(a, b, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    y = acc + bias[None, :]
    y = y * scaling
    y = tl.minimum(tl.maximum(y, ht_min), ht_max)
    y = 0.5 * y * (1.0 + tl.erf(y * 0.7071067811865476))

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, y.to(out_ptr.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


class Model(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)
        self.gelu = nn.GELU()

    def forward(self, x):
        orig_shape = x.shape
        K = orig_shape[-1]
        x2 = x.reshape(-1, K)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        M = x2.shape[0]
        w = self.gemm.weight
        b = self.gemm.bias
        N = w.shape[0]
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        _linear_scale_hardtanh_gelu_kernel[grid](
            x2, w, b, out,
            M, N, K,
            x2.stride(0), x2.stride(1),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1),
            float(self.scaling_factor),
            float(self.hardtanh.min_val), float(self.hardtanh.max_val),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_M=8,
            num_warps=8, num_stages=3,
        )
        return out.reshape(*orig_shape[:-1], N)
