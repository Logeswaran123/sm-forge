import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fp8_linear_kernel(
    x_ptr, w_ptr, out_ptr, in_scale_ptr, w_scale_ptr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        a_ptrs = x_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk
        a_mask = (offs_m[:, None] < M) & (kk[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        # B tile [BLOCK_K, BLOCK_N] from weight [N, K]
        b_ptrs = w_ptr + kk[:, None] * stride_wk + offs_n[None, :] * stride_wn
        b_mask = (kk[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    s = tl.load(in_scale_ptr).to(tl.float32) * tl.load(w_scale_ptr).to(tl.float32)
    acc = acc * s

    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=o_mask)


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        weight = 0.05 * torch.randn(out_features, in_features)
        scale = weight.abs().amax() / 448.0
        self.register_buffer("weight", (weight / scale).to(torch.float8_e4m3fn))
        self.register_buffer("weight_scale", scale)
        self.register_buffer("input_scale", torch.tensor(0.05))

    def forward(self, x):
        K = x.shape[-1]
        lead = x.shape[:-1]
        x2 = x.reshape(-1, K)
        if x2.dtype != torch.float8_e4m3fn:
            x2 = x2.to(torch.float8_e4m3fn)
        x2 = x2.contiguous()
        w = self.weight.contiguous()
        M = x2.shape[0]
        N = w.shape[0]
        out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

        in_scale = self.input_scale.to(device=x.device, dtype=torch.float32).contiguous()
        w_scale = self.weight_scale.to(device=x.device, dtype=torch.float32).contiguous()

        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        fp8_linear_kernel[grid](
            x2, w, out, in_scale, w_scale,
            M, N, K,
            x2.stride(0), x2.stride(1),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_M=8,
            num_warps=8, num_stages=3,
        )
        return out.reshape(*lead, N)
