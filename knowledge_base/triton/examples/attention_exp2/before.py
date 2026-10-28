import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(
    Q, K, V, Out,
    sqb, sqh, sqm, sqd,
    skb, skh, skn, skd,
    svb, svh, svn, svd,
    sob, soh, som, sod,
    H, GROUP, LQ, LK,
    qk_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    hk = h // GROUP

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_D)

    q_ptrs = Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < LQ, other=0.0)

    k_base = K + b * skb + hk * skh
    v_base = V + b * svb + hk * svh

    m_i = tl.full([BLOCK_M], -1.0e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_D], dtype=tl.float32)

    hi = tl.minimum((pid_m + 1) * BLOCK_M, LK)
    for start_n in range(0, hi, BLOCK_N):
        cur_n = start_n + offs_n
        n_mask = cur_n < LK
        k = tl.load(k_base + cur_n[None, :] * skn + offs_d[:, None] * skd,
                    mask=n_mask[None, :], other=0.0)  # [D, BN]
        s = tl.dot(q, k) * qk_scale
        valid = (cur_n[None, :] <= offs_m[:, None]) & n_mask[None, :]
        s = tl.where(valid, s, -1.0e30)

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        p = tl.where(valid, p, 0.0)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = tl.load(v_base + cur_n[:, None] * svn + offs_d[None, :] * svd,
                    mask=n_mask[:, None], other=0.0)  # [BN, D]
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out = acc / l_safe[:, None]
    o_ptrs = Out + b * sob + h * soh + offs_m[:, None] * som + offs_d[None, :] * sod
    tl.store(o_ptrs, out.to(Out.dtype.element_ty), mask=offs_m[:, None] < LQ)


class Model(nn.Module):
    def forward(self, q, k, v):
        B, H, LQ, D = q.shape
        HK = k.shape[1]
        LK = k.shape[2]
        group = H // HK
        o = torch.empty((B, H, LQ, D), device=q.device, dtype=q.dtype)
        BLOCK_M = 64
        BLOCK_N = 64
        qk_scale = 1.0 / math.sqrt(D)
        grid = (triton.cdiv(LQ, BLOCK_M), B * H)
        _attn_fwd[grid](
            q, k, v, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            H, group, LQ, LK,
            qk_scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_D=D,
            num_warps=4, num_stages=2,
        )
        return o
