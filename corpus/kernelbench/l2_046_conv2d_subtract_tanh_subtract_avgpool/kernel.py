import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _tanh(x):
    # stable: large +x -> exp=inf -> 1 ; large -x -> exp=0 -> -1
    e = tl.exp(2.0 * x)
    return 1.0 - 2.0 / (e + 1.0)


@triton.jit
def conv_sub_tanh_sub_pool_kernel(
    x_ptr,          # NHWC input [N, H, W, C]
    w_ptr,          # weights [KH, KW, C, CO]
    b_ptr,          # bias [CO]
    out_ptr,        # NCHW output [N, CO, PH, PW]
    N, C, H, W, CO, PH, PW,
    sub1, sub2,
    KH: tl.constexpr, KW: tl.constexpr, P: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = N * PH * PW
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < CO

    nb = offs_m // (PH * PW)
    rem = offs_m % (PH * PW)
    ph = rem // PW
    pw = rem % PW

    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)

    pool = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for dy in tl.static_range(P):
        for dx in tl.static_range(P):
            oh = ph * P + dy
            ow = pw * P + dx
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for kh in tl.static_range(KH):
                for kw in tl.static_range(KW):
                    ih = oh + kh
                    iw = ow + kw
                    base = ((nb.to(tl.int64) * H + ih) * W + iw) * C
                    for c0 in range(0, C, BLOCK_K):
                        offs_k = c0 + tl.arange(0, BLOCK_K)
                        k_mask = offs_k < C
                        a = tl.load(
                            x_ptr + base[:, None] + offs_k[None, :],
                            mask=m_mask[:, None] & k_mask[None, :],
                            other=0.0,
                        )
                        w_row = (kh * KW + kw) * C + offs_k
                        b = tl.load(
                            w_ptr + w_row[:, None] * CO + offs_n[None, :],
                            mask=k_mask[:, None] & n_mask[None, :],
                            other=0.0,
                        )
                        acc = tl.dot(a, b, acc)
            v = acc + bias[None, :] - sub1
            pool += _tanh(v) - sub2
    pool = pool / (P * P)

    out_off = ((nb.to(tl.int64)[:, None] * CO + offs_n[None, :]) * PH + ph[:, None]) * PW + pw[:, None]
    tl.store(
        out_ptr + out_off,
        pool.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)
        self.kernel_size_pool = kernel_size_pool

    def forward(self, x):
        w = self.conv.weight
        b = self.conv.bias
        N, C, H, W = x.shape
        CO, _, KH, KW = w.shape
        P = self.kernel_size_pool
        if isinstance(P, (tuple, list)):
            P = P[0]
        OH = H - KH + 1
        OW = W - KW + 1
        PH = OH // P
        PW = OW // P

        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        w_t = w.to(x.dtype).permute(2, 3, 1, 0).contiguous()  # [KH, KW, C, CO]
        out = torch.empty((N, CO, PH, PW), device=x.device, dtype=x.dtype)

        M = N * PH * PW
        if M == 0:
            return out
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(CO, BLOCK_N))
        conv_sub_tanh_sub_pool_kernel[grid](
            x_nhwc, w_t, b, out,
            N, C, H, W, CO, PH, PW,
            float(self.subtract1_value), float(self.subtract2_value),
            KH=KH, KW=KW, P=P,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )
        return out
