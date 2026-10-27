import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_gelu_partial_pool_kernel(
    x_ptr, w_ptr, b_ptr, part_ptr,
    C_in, H, W, C_out, OH, OW, NPB,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_p = tl.program_id(1)

    P = OH * OW
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    mask_p = offs_p < P
    oh = offs_p // OW
    ow = offs_p % OW

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C_out

    acc = tl.zeros((BLOCK_P, BLOCK_C), dtype=tl.float32)

    x_base = x_ptr + pid_b.to(tl.int64) * C_in * H * W
    for ci in range(0, C_in):
        x_c = x_base + ci * H * W
        for kh in tl.static_range(KH):
            for kw in tl.static_range(KW):
                xv = tl.load(x_c + (oh + kh) * W + (ow + kw), mask=mask_p, other=0.0).to(tl.float32)
                w_off = ((offs_c * C_in + ci) * KH + kh) * KW + kw
                wv = tl.load(w_ptr + w_off, mask=mask_c, other=0.0).to(tl.float32)
                acc += xv[:, None] * wv[None, :]

    bias = tl.load(b_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # exact GELU
    g = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    g = tl.where(mask_p[:, None] & mask_c[None, :], g, 0.0)

    s = tl.sum(g, axis=0)  # [BLOCK_C]
    out_off = (pid_b * NPB + pid_p) * C_out + offs_c
    tl.store(part_ptr + out_off, s, mask=mask_c)


@triton.jit
def finalize_pool_kernel(
    part_ptr, out_ptr, C_out, NPB, inv_count,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C_out
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    base = part_ptr + pid_b * NPB * C_out
    for i in range(0, NPB):
        acc += tl.load(base + i * C_out + offs_c, mask=mask_c, other=0.0)
    res = acc * inv_count
    tl.store(out_ptr + pid_b * C_out + offs_c, res.to(out_ptr.dtype.element_ty), mask=mask_c)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()

        B, C_in, H, W = x.shape
        C_out, _, KH, KW = w.shape
        OH = H - KH + 1
        OW = W - KW + 1
        P = OH * OW

        BLOCK_P = 128
        BLOCK_C = triton.next_power_of_2(C_out)
        NPB = triton.cdiv(P, BLOCK_P)

        part = torch.empty((B, NPB, C_out), device=x.device, dtype=torch.float32)
        conv_gelu_partial_pool_kernel[(B, NPB)](
            x, w, b, part,
            C_in, H, W, C_out, OH, OW, NPB,
            KH=KH, KW=KW, BLOCK_P=BLOCK_P, BLOCK_C=BLOCK_C,
            num_warps=4,
        )

        out = torch.empty((B, C_out), device=x.device, dtype=x.dtype)
        FC = min(BLOCK_C, 64)
        finalize_pool_kernel[(B, triton.cdiv(C_out, FC))](
            part, out, C_out, NPB, 1.0 / float(P),
            BLOCK_C=FC, num_warps=2,
        )
        return out
