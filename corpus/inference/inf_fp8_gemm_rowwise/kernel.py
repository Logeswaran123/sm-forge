import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fp8_scaled_gemm_kernel(
    x_ptr, w_ptr, xs_ptr, ws_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # B tile [BLOCK_K, BLOCK_N] read from weight [N, K] (i.e. weight^T)
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + offs_k) < K
        a = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(w_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    xs = tl.load(xs_ptr + offs_m, mask=mask_m, other=0.0)
    ws = tl.load(ws_ptr + offs_n, mask=mask_n, other=0.0)
    out = acc * xs[:, None] * ws[None, :]

    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(o_ptrs, out.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        weight = 0.05 * torch.randn(out_features, in_features)
        scale = weight.abs().amax(dim=1) / 448.0
        self.register_buffer("weight", (weight / scale[:, None]).to(torch.float8_e4m3fn))
        self.register_buffer("weight_scale", scale)

    def forward(self, x, x_scale):
        x = x.contiguous()
        x_scale = x_scale.contiguous().to(torch.float32)
        w = self.weight.contiguous()
        ws = self.weight_scale.contiguous().to(torch.float32)

        M, K = x.shape
        N = w.shape[0]
        out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        fp8_scaled_gemm_kernel[grid](
            x, w, x_scale, ws, out,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=3,
        )
        return out
