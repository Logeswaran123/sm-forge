import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_bn_gelu_relu_kernel(
    x_ptr, w_ptr, b_ptr,
    gamma_ptr, beta_ptr, mean_ptr, var_ptr,
    out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    eps,
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
        a = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        )
        b = tl.load(
            w_ptr + offs_n[None, :] * stride_wn + kk[:, None] * stride_wk,
            mask=mask_n[None, :] & mask_k[:, None], other=0.0,
        )
        b = b.to(a.dtype)
        acc = tl.dot(a, b, acc)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)

    scale = gamma / tl.sqrt(var + eps)
    y = acc + bias[None, :]
    y = (y - mean[None, :]) * scale[None, :] + beta[None, :]
    # exact GELU
    y = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    # ReLU
    y = tl.maximum(y, 0.0)

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        y.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


class Model(nn.Module):
    """
    GEMM + BatchNorm (eval) + GELU + ReLU, fused in a single Triton kernel.
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x):
        K = x.shape[-1]
        x2 = x.reshape(-1, K)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        M = x2.shape[0]
        w = self.gemm.weight
        N = w.shape[0]
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 128, 32, 8
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        bn = self.batch_norm
        fused_linear_bn_gelu_relu_kernel[grid](
            x2, w, self.gemm.bias,
            bn.weight, bn.bias, bn.running_mean, bn.running_var,
            out,
            M, N, K,
            x2.stride(0), x2.stride(1),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1),
            float(bn.eps),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
            num_warps=8, num_stages=3,
        )
        return out.reshape(*x.shape[:-1], N)


batch_size = 16384
in_features = 4096
out_features = 4096


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]
