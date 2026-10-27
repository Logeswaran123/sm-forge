import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_swish_scale_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    scale,
    HAS_BIAS: tl.constexpr,
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
        k = k0 + offs_k
        mask_k = k < K
        a = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        )
        # W is [N, K]; load a [BLOCK_K, BLOCK_N] tile of W^T
        w = tl.load(
            w_ptr + offs_n[None, :] * stride_wn + k[:, None] * stride_wk,
            mask=mask_n[None, :] & mask_k[:, None], other=0.0,
        )
        w = w.to(a.dtype)
        acc = tl.dot(a, w, acc)

    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    y = acc * tl.sigmoid(acc)
    y = y * scale

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        y.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


class Model(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        w = self.matmul.weight
        b = self.matmul.bias
        K = x.shape[-1]
        lead = x.shape[:-1]
        x2 = x.reshape(-1, K)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        M = x2.shape[0]
        N = w.shape[0]
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        BLOCK_M = 16 if M <= 16 else 64
        BLOCK_N = 128
        BLOCK_K = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        has_bias = b is not None
        linear_swish_scale_kernel[grid](
            x2, w, b if has_bias else w, out,
            M, N, K,
            x2.stride(0), x2.stride(1),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1),
            float(self.scaling_factor),
            HAS_BIAS=has_bias,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )
        return out.reshape(*lead, N)
