import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_bn_swish_kernel(
    x_ptr, w_ptr, lin_b_ptr,
    bn_w_ptr, bn_b_ptr, bn_mean_ptr, bn_var_ptr,
    bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    bias_stride,
    eps, inv_div,
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
    # B tile [BLOCK_K, BLOCK_N] from W[N, K]
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + offs_k) < K
        a = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(w_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, b.to(a.dtype), acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    lin_b = tl.load(lin_b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    gamma = tl.load(bn_w_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    beta = tl.load(bn_b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    mean = tl.load(bn_mean_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    var = tl.load(bn_var_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    extra_b = tl.load(bias_ptr + offs_n * bias_stride, mask=mask_n, other=0.0).to(tl.float32)

    scale = gamma / tl.sqrt(var + eps)
    shift = beta - mean * scale

    z = acc + lin_b[None, :]
    z = z * scale[None, :] + shift[None, :]
    z = z + extra_b[None, :]
    z = z * inv_div
    y = z * (1.0 / (1.0 + tl.exp(-z)))

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, y.to(out_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


class Model(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1,
                 bias_shape=(1,), divide_value=1.0):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value

    def forward(self, x):
        orig_shape = x.shape
        K = orig_shape[-1]
        x2 = x.reshape(-1, K)
        M = x2.shape[0]
        w = self.matmul.weight
        N = w.shape[0]
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        bias = self.bias.reshape(-1)
        bias_stride = 0 if bias.numel() == 1 else bias.stride(0)

        lin_b = self.matmul.bias
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        fused_linear_bn_swish_kernel[grid](
            x2, w, lin_b,
            self.bn.weight, self.bn.bias, self.bn.running_mean, self.bn.running_var,
            bias, out,
            M, N, K,
            x2.stride(0), x2.stride(1),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1),
            bias_stride,
            float(self.bn.eps), 1.0 / float(self.divide_value),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=8,
            num_warps=8, num_stages=3,
        )
        return out.reshape(*orig_shape[:-1], N)
