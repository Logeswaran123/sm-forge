import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_div_gelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    inv_div,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

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
        # W tile laid out as [BLOCK_K, BLOCK_N] (W is [N, K])
        b = tl.load(
            w_ptr + offs_n[None, :] * stride_wn + kk[:, None] * stride_wk,
            mask=mask_n[None, :] & mask_k[:, None], other=0.0,
        )
        b = b.to(a.dtype)
        acc = tl.dot(a, b, acc)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    y = (acc + bias[None, :]) * inv_div
    # exact GELU: 0.5 * y * (1 + erf(y / sqrt(2)))
    y = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, y.to(out_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


class Model(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor

    def forward(self, x):
        orig_shape = x.shape
        K = orig_shape[-1]
        x2 = x.reshape(-1, K)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        M = x2.shape[0]
        w = self.linear.weight
        b = self.linear.bias
        N = w.shape[0]

        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        linear_div_gelu_kernel[grid](
            x2, w, b, out,
            M, N, K,
            x2.stride(0), x2.stride(1),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1),
            1.0 / float(self.divisor),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )
        return out.reshape(*orig_shape[:-1], N)
