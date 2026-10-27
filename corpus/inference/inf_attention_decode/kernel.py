import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gqa_decode_attn_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    sqb, sqh, sqm, sqd,
    skb, skh, skn, skd,
    svb, svh, svn, svd,
    sob, soh, som, sod,
    Hkv, G, Lq, S, D,
    qk_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    b = pid_bh // Hkv
    kh = pid_bh % Hkv

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < G * Lq
    g = offs_m // Lq
    l = offs_m % Lq
    h = kh * G + g

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    q_ptrs = (q_ptr + b * sqb + h[:, None] * sqh + l[:, None] * sqm
              + offs_d[None, :] * sqd)
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    k_base = k_ptr + b * skb + kh * skh
    v_base = v_ptr + b * svb + kh * svh

    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        kv_mask = mask_n[:, None] & mask_d[None, :]
        k = tl.load(k_base + offs_n[:, None] * skn + offs_d[None, :] * skd,
                    mask=kv_mask, other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale  # log2 domain
        s = tl.where(mask_n[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(v_base + offs_n[:, None] * svn + offs_d[None, :] * svd,
                    mask=kv_mask, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    out = acc / l_i[:, None]
    o_ptrs = (o_ptr + b * sob + h[:, None] * soh + l[:, None] * som
              + offs_d[None, :] * sod)
    tl.store(o_ptrs, out.to(o_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_d[None, :])


class Model(nn.Module):
    def forward(self, q, k_cache, v_cache):
        B, Hq, Lq, D = q.shape
        _, Hkv, S, _ = k_cache.shape
        G = Hq // Hkv
        o = torch.empty((B, Hq, Lq, D), dtype=q.dtype, device=q.device)

        BLOCK_M = 16
        BLOCK_N = 64
        BLOCK_D = max(16, triton.next_power_of_2(D))
        qk_scale = (1.0 / math.sqrt(D)) * 1.4426950408889634

        grid = (B * Hkv, triton.cdiv(G * Lq, BLOCK_M))
        _gqa_decode_attn_kernel[grid](
            q, k_cache, v_cache, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            Hkv, G, Lq, S, D,
            qk_scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,
        )
        return o
