import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    pos_ptr, cos_ptr, sin_ptr, x_ptr, out_ptr,
    H, HALF,
    stride_xt, stride_xh,
    stride_ot, stride_oh,
    stride_c,
    BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0)
    pos = tl.load(pos_ptr + t).to(tl.int64)

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HALF
    mask_h = offs_h < H

    c = tl.load(cos_ptr + pos * stride_c + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + pos * stride_c + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    c = c[None, :]
    s = s[None, :]

    mask = mask_h[:, None] & mask_d[None, :]
    t64 = t.to(tl.int64)
    x_base = x_ptr + t64 * stride_xt + offs_h[:, None] * stride_xh + offs_d[None, :]
    x1 = tl.load(x_base, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_base + HALF, mask=mask, other=0.0).to(tl.float32)

    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s

    o_base = out_ptr + t64 * stride_ot + offs_h[:, None] * stride_oh + offs_d[None, :]
    tl.store(o_base, o1.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(o_base + HALF, o2.to(out_ptr.dtype.element_ty), mask=mask)


def _rope(positions, x, cos, sin):
    T, H, D = x.shape
    half = D // 2
    x = x.contiguous()
    out = torch.empty_like(x)
    if T == 0:
        return out
    BLOCK_H = triton.next_power_of_2(H)
    BLOCK_D = triton.next_power_of_2(half)
    _rope_kernel[(T,)](
        positions, cos, sin, x, out,
        H, half,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        cos.stride(0),
        BLOCK_H=BLOCK_H, BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return out


class Model(nn.Module):
    def __init__(self, head_dim, max_position, base):
        super().__init__()
        inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        freqs = torch.outer(torch.arange(max_position, dtype=torch.float32), inv_freq)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, positions, q, k):
        positions = positions.contiguous().view(-1)
        cos = self.cos.contiguous()
        sin = self.sin.contiguous()
        q_out = _rope(positions, q, cos, sin)
        k_out = _rope(positions, k, cos, sin)
        return q_out, k_out
