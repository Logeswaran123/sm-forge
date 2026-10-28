import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def paged_attn_decode_kernel(
    q_ptr, k_ptr, v_ptr, bt_ptr, sl_ptr, o_ptr,
    sm_scale, MB,
    stride_qb, stride_qh,
    stride_kb, stride_ks, stride_kh,
    stride_vb, stride_vs, stride_vh,
    stride_btb,
    stride_ob, stride_oh,
    G, BS, D,
    GP: tl.constexpr, BSP: tl.constexpr, DP: tl.constexpr,
):
    b = tl.program_id(0)
    kvh = tl.program_id(1)

    offs_g = tl.arange(0, GP)
    offs_n = tl.arange(0, BSP)
    offs_d = tl.arange(0, DP)
    gmask = offs_g < G
    nmask = offs_n < BS
    dmask = offs_d < D

    heads = kvh * G + offs_g
    q = tl.load(
        q_ptr + b * stride_qb + heads[:, None] * stride_qh + offs_d[None, :],
        mask=gmask[:, None] & dmask[None, :], other=0.0,
    ).to(tl.float32)

    seq_len = tl.load(sl_ptr + b)
    num_pages = (seq_len + BS - 1) // BS
    num_pages = tl.minimum(num_pages, MB)

    m = tl.full([GP], -1e30, dtype=tl.float32)
    l_sum = tl.zeros([GP], dtype=tl.float32)
    acc = tl.zeros([GP, DP], dtype=tl.float32)

    kv_mask = nmask[:, None] & dmask[None, :]
    for p in range(0, num_pages):
        blk = tl.load(bt_ptr + b * stride_btb + p).to(tl.int64)
        k = tl.load(
            k_ptr + blk * stride_kb + offs_n[:, None] * stride_ks
            + kvh * stride_kh + offs_d[None, :],
            mask=kv_mask, other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptr + blk * stride_vb + offs_n[:, None] * stride_vs
            + kvh * stride_vh + offs_d[None, :],
            mask=kv_mask, other=0.0,
        ).to(tl.float32)

        s = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * sm_scale  # [GP, BSP]
        valid = ((p * BS + offs_n) < seq_len) & nmask
        s = tl.where(valid[None, :], s, float("-inf"))

        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp(m - m_new)
        pr = tl.exp(s - m_new[:, None])
        l_sum = l_sum * alpha + tl.sum(pr, axis=1)
        acc = acc * alpha[:, None] + tl.sum(pr[:, :, None] * v[None, :, :], axis=1)
        m = m_new

    out = acc / l_sum[:, None]
    tl.store(
        o_ptr + b * stride_ob + heads[:, None] * stride_oh + offs_d[None, :],
        out.to(o_ptr.dtype.element_ty),
        mask=gmask[:, None] & dmask[None, :],
    )


class Model(nn.Module):
    def forward(self, q, key_cache, value_cache, block_tables, seq_lens):
        batch, heads, dim = q.shape
        q = q.contiguous()
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()
        block_tables = block_tables.contiguous()
        seq_lens = seq_lens.contiguous()

        _, bs, kv_heads, _ = key_cache.shape
        G = heads // kv_heads
        MB = block_tables.shape[1]

        out = torch.empty((batch, heads, dim), dtype=q.dtype, device=q.device)

        GP = triton.next_power_of_2(G)
        BSP = triton.next_power_of_2(bs)
        DP = triton.next_power_of_2(dim)

        grid = (batch, kv_heads)
        paged_attn_decode_kernel[grid](
            q, key_cache, value_cache, block_tables, seq_lens, out,
            1.0 / (dim ** 0.5), MB,
            q.stride(0), q.stride(1),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            block_tables.stride(0),
            out.stride(0), out.stride(1),
            G, bs, dim,
            GP=GP, BSP=BSP, DP=DP,
            num_warps=4,
        )
        return out
