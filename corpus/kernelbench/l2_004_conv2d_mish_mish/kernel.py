import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mish(x):
    # mish(x) = x * tanh(softplus(x)); tanh(log(1+e^x)) = n / (n + 2), n = e^x (e^x + 2)
    t = tl.exp(tl.minimum(x, 20.0))
    n = t * (t + 2.0)
    return x * (n / (n + 2.0))


@triton.jit
def conv_mish_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    NB, C, H, W, OC, OH, OW, K,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = NB * OH * OW
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    ow = offs_m % OW
    t = offs_m // OW
    oh = t % OH
    n = t // OH
    x_base = n * (C * H * W) + oh * W + ow

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        kw = offs_k % KW
        kh = (offs_k // KW) % KH
        c = offs_k // (KH * KW)
        k_off = c * (H * W) + kh * W + kw

        a = tl.load(
            x_ptr + x_base[:, None] + k_off[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            w_ptr + offs_n[None, :] * K + offs_k[:, None],
            mask=n_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        a = a.to(tl.float16)
        b = b.to(tl.float16)
        acc = tl.dot(a, b, acc)

    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    y = _mish(acc)
    y = _mish(y)

    out_off = (n * (OC * OH * OW) + oh * OW + ow)[:, None] + (offs_n * (OH * OW))[None, :]
    tl.store(out_ptr + out_off, y.to(out_ptr.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        NB, C, H, W = x.shape
        OC, _, KH, KW = w.shape
        OH = H - KH + 1
        OW = W - KW + 1
        K = C * KH * KW
        out = torch.empty((NB, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        M = NB * OH * OW
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(OC, BLOCK_N))
        conv_mish_mish_kernel[grid](
            x, w, b, out,
            NB, C, H, W, OC, OH, OW, K,
            KH=KH, KW=KW,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=3,
        )
        return out
