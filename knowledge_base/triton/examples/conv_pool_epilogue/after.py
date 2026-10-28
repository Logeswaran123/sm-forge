import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_gelu_pool_kernel(
    x_ptr, w_ptr, b_ptr, part_ptr,
    C, H, W, OC, OH, OW, K, NPB,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    n_idx = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    ohw = OH * OW
    offs_p = pid_p * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_p = offs_p < ohw
    mask_n = offs_n < OC
    oh = offs_p // OW
    ow = offs_p % OW
    x_base = n_idx.to(tl.int64) * (C * H * W) + oh * W + ow

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        c = offs_k // (KH * KW)
        r = offs_k % (KH * KW)
        kh = r // KW
        kw = r % KW
        k_off = c * (H * W) + kh * W + kw

        a = tl.load(x_ptr + x_base[:, None] + k_off[None, :],
                    mask=mask_p[:, None] & mask_k[None, :], other=0.0)
        wv = tl.load(w_ptr + offs_n[None, :] * K + offs_k[:, None],
                     mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, wv.to(a.dtype))

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    acc = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    acc = tl.where(mask_p[:, None], acc, 0.0)
    part = tl.sum(acc, axis=0)
    tl.store(part_ptr + (n_idx * NPB + pid_p) * OC + offs_n, part, mask=mask_n)


@triton.jit
def finalize_kernel(part_ptr, out_ptr, OC, NPB, inv_hw, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr):
    n_idx = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_C)
    offs_p = tl.arange(0, BLOCK_P)
    mask_c = offs_c < OC
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for start in range(0, NPB, BLOCK_P):
        rows = start + offs_p
        mask = (rows < NPB)[:, None] & mask_c[None, :]
        vals = tl.load(part_ptr + (n_idx * NPB + rows)[:, None] * OC + offs_c[None, :], mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    tl.store(out_ptr + n_idx * OC + offs_c, (acc * inv_hw).to(out_ptr.dtype.element_ty), mask=mask_c)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
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
        out = torch.empty((NB, OC), device=x.device, dtype=x.dtype)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 16
        NPB = triton.cdiv(OH * OW, BLOCK_M)
        part = torch.empty((NB, NPB, OC), device=x.device, dtype=torch.float32)
        grid = (NB, NPB, triton.cdiv(OC, BLOCK_N))
        conv_gelu_pool_kernel[grid](
            x, w, b, part,
            C, H, W, OC, OH, OW, K, NPB,
            KH=KH, KW=KW,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        finalize_kernel[(NB,)](
            part, out, OC, NPB, 1.0 / (OH * OW),
            BLOCK_P=64, BLOCK_C=triton.next_power_of_2(OC), num_warps=4,
        )
        return out
