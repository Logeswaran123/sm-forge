import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _tanh(x):
    # tanh(x) = 1 - 2 / (exp(2x) + 1); saturates correctly to +/-1
    return 1.0 - 2.0 / (tl.exp(2.0 * x) + 1.0)


@triton.jit
def conv_min_tanh_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    C_in, H, W, C_out, OH, OW,
    KS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_p = tl.program_id(0)
    n = tl.program_id(1).to(tl.int64)

    P = OH * OW
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    mask_p = offs_p < P
    oh = offs_p // OW
    ow = offs_p % OW

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C_out

    acc = tl.zeros((BLOCK_P, BLOCK_C), dtype=tl.float32)

    x_base = x_ptr + n * C_in * H * W
    for ci in range(0, C_in):
        x_ci = x_base + ci.to(tl.int64) * H * W
        w_ci = w_ptr + ci * KS * KS
        for kh in tl.static_range(KS):
            for kw in tl.static_range(KS):
                xv = tl.load(x_ci + (oh + kh) * W + (ow + kw), mask=mask_p, other=0.0).to(tl.float32)
                wv = tl.load(w_ci + offs_c * (C_in * KS * KS) + kh * KS + kw,
                             mask=mask_c, other=0.0).to(tl.float32)
                acc += xv[:, None] * wv[None, :]

    bias = tl.load(b_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    acc = tl.where(mask_c[None, :], acc, float("inf"))
    m = tl.min(acc, axis=1)
    y = _tanh(_tanh(m))

    tl.store(out_ptr + n * P + offs_p, y.to(out_ptr.dtype.element_ty), mask=mask_p)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        N, C_in, H, W = x.shape
        C_out = w.shape[0]
        KS = w.shape[2]
        OH = H - KS + 1
        OW = W - KS + 1
        out = torch.empty((N, 1, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_P = 128
        BLOCK_C = max(16, triton.next_power_of_2(C_out))
        grid = (triton.cdiv(OH * OW, BLOCK_P), N)
        conv_min_tanh_kernel[grid](
            x, w, b, out,
            C_in, H, W, C_out, OH, OW,
            KS=KS, BLOCK_P=BLOCK_P, BLOCK_C=BLOCK_C,
            num_warps=4,
        )
        return out
