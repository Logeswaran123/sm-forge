import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_gn_lrelu_kernel(
    x_ptr, w_ptr, b_ptr, gw_ptr, gb_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    eps, neg_slope,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + offs_k
        k_mask = kk < K
        a = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        )
        b = tl.load(
            w_ptr + offs_n[None, :] * stride_wn + kk[:, None] * stride_wk,
            mask=n_mask[None, :] & k_mask[:, None], other=0.0,
        )
        acc = tl.dot(a, b.to(a.dtype), acc)

    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # Group norm over groups of GS contiguous channels (fully inside the tile)
    NG: tl.constexpr = BN // GS
    v = tl.reshape(acc, (BM, NG, GS))
    mean = tl.sum(v, axis=2) / GS
    d = v - tl.expand_dims(mean, 2)
    var = tl.sum(d * d, axis=2) / GS
    rstd = 1.0 / tl.sqrt(var + eps)
    y = d * tl.expand_dims(rstd, 2)
    y = tl.reshape(y, (BM, BN))

    gw = tl.load(gw_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    gb = tl.load(gb_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    y = y * gw[None, :] + gb[None, :]

    y = tl.where(y >= 0, y, y * neg_slope)
    y = y + y

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        y.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        w = self.fc.weight
        N = w.shape[0]
        GS = N // self.num_groups
        # GS must be a power of two for in-tile reshape; BN multiple of GS
        BN = max(GS, 128)
        BM = 64
        BK = 64
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        fused_linear_gn_lrelu_kernel[grid](
            x, w, self.fc.bias, self.gn.weight, self.gn.bias, out,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1),
            float(self.eps), float(self.negative_slope),
            BM=BM, BN=BN, BK=BK, GS=GS,
            num_warps=4, num_stages=3,
        )
        return out


batch_size = 1024
input_size = 8192
hidden_size = 8192
num_groups = 512


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, num_groups]
