import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def paged_attn_decode_kernel(
    q_ptr, k_ptr, v_ptr, bt_ptr, sl_ptr, m_ws, l_ws, acc_ws,
    sm_scale, MB, PAGES_PER_SPLIT,
    stride_qb, stride_qh,
    stride_kb, stride_ks, stride_kh,
    stride_vb, stride_vs, stride_vh,
    stride_btb,
    G, BS, D, H, SPLITS,
    GP: tl.constexpr, BSP: tl.constexpr, DP: tl.constexpr,
):
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    split = tl.program_id(2)

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
    first = split * PAGES_PER_SPLIT
    last = tl.minimum(first + PAGES_PER_SPLIT, num_pages)
    for p in range(first, last):
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

    ws = (b * H + heads) * SPLITS + split
    tl.store(m_ws + ws, m, mask=gmask)
    tl.store(l_ws + ws, l_sum, mask=gmask)
    tl.store(acc_ws + ws[:, None] * D + offs_d[None, :], acc, mask=gmask[:, None] & dmask[None, :])


@triton.jit
def combine_kernel(
    m_ws, l_ws, acc_ws, o_ptr, stride_ob, stride_oh, H, D, SPLITS,
    SP: tl.constexpr, DP: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_s = tl.arange(0, SP)
    offs_d = tl.arange(0, DP)
    smask = offs_s < SPLITS
    dmask = offs_d < D
    ws = (b * H + h) * SPLITS + offs_s
    m = tl.load(m_ws + ws, mask=smask, other=-1e30)
    l_sum = tl.load(l_ws + ws, mask=smask, other=0.0)
    acc = tl.load(acc_ws + ws[:, None] * D + offs_d[None, :], mask=smask[:, None] & dmask[None, :], other=0.0)
    m_all = tl.max(m, axis=0)
    scale = tl.exp(m - m_all)
    out = tl.sum(acc * scale[:, None], axis=0) / tl.sum(l_sum * scale, axis=0)
    tl.store(o_ptr + b * stride_ob + h * stride_oh + offs_d, out.to(o_ptr.dtype.element_ty), mask=dmask)


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

        SPLITS = 16
        pages_per_split = triton.cdiv(MB, SPLITS)
        m_ws = torch.empty((batch, heads, SPLITS), dtype=torch.float32, device=q.device)
        l_ws = torch.empty((batch, heads, SPLITS), dtype=torch.float32, device=q.device)
        acc_ws = torch.empty((batch, heads, SPLITS, dim), dtype=torch.float32, device=q.device)
        paged_attn_decode_kernel[(batch, kv_heads, SPLITS)](
            q, key_cache, value_cache, block_tables, seq_lens, m_ws, l_ws, acc_ws,
            1.0 / (dim ** 0.5), MB, pages_per_split,
            q.stride(0), q.stride(1),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            block_tables.stride(0),
            G, bs, dim, heads, SPLITS,
            GP=GP, BSP=BSP, DP=DP,
            num_warps=4,
        )
        combine_kernel[(batch, heads)](
            m_ws, l_ws, acc_ws, out, out.stride(0), out.stride(1), heads, dim, SPLITS,
            SP=triton.next_power_of_2(SPLITS), DP=DP,
        )
        return out
