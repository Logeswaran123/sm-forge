import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _qk_kernel(q_ptr, k_ptr, s_ptr, M, N, D, scale,
               sq_b, sq_m, sq_d,
               sk_b, sk_n, sk_d,
               ss_b, ss_m, ss_n,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    bh = tl.program_id(2).to(tl.int64)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    q_base = q_ptr + bh * sq_b
    k_base = k_ptr + bh * sk_b
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, D, BK):
        kk = k0 + offs_k
        q = tl.load(q_base + offs_m[:, None] * sq_m + kk[None, :] * sq_d,
                    mask=(offs_m[:, None] < M) & (kk[None, :] < D), other=0.0)
        kt = tl.load(k_base + offs_n[None, :] * sk_n + kk[:, None] * sk_d,
                     mask=(offs_n[None, :] < N) & (kk[:, None] < D), other=0.0)
        acc += tl.dot(q, kt)
    acc = acc * scale
    s_base = s_ptr + bh * ss_b
    tl.store(s_base + offs_m[:, None] * ss_m + offs_n[None, :] * ss_n, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _softmax_kernel(s_ptr, p_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(s_ptr + row * N + offs, mask=mask, other=float("-inf"))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    ssum = tl.sum(e, axis=0)
    p = e / ssum
    tl.store(p_ptr + row * N + offs, p.to(p_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _pv_kernel(p_ptr, v_ptr, o_ptr, M, N, D,
               sp_b, sp_m, sp_n,
               sv_b, sv_n, sv_d,
               so_b, so_m, so_d,
               BM: tl.constexpr, BD: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    bh = tl.program_id(2).to(tl.int64)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = pid_d * BD + tl.arange(0, BD)
    offs_k = tl.arange(0, BK)
    p_base = p_ptr + bh * sp_b
    v_base = v_ptr + bh * sv_b
    acc = tl.zeros((BM, BD), dtype=tl.float32)
    for n0 in range(0, N, BK):
        nn_ = n0 + offs_k
        p = tl.load(p_base + offs_m[:, None] * sp_m + nn_[None, :] * sp_n,
                    mask=(offs_m[:, None] < M) & (nn_[None, :] < N), other=0.0)
        v = tl.load(v_base + nn_[:, None] * sv_n + offs_d[None, :] * sv_d,
                    mask=(nn_[:, None] < N) & (offs_d[None, :] < D), other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
    o_base = o_ptr + bh * so_b
    tl.store(o_base + offs_m[:, None] * so_m + offs_d[None, :] * so_d,
             acc.to(o_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_d[None, :] < D))


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, H, M, D = Q.shape
        N = K.shape[2]
        Dv = V.shape[3]
        q = Q.contiguous().reshape(B * H, M, D)
        k = K.contiguous().reshape(B * H, N, D)
        v = V.contiguous().reshape(B * H, N, Dv)
        BH = B * H
        dev = Q.device

        S = torch.empty((BH, M, N), device=dev, dtype=torch.float32)
        P = torch.empty((BH, M, N), device=dev, dtype=V.dtype)
        O = torch.empty((BH, M, Dv), device=dev, dtype=Q.dtype)

        scale = 1.0 / math.sqrt(D)

        BM, BN, BK = 64, 64, 64
        grid1 = (triton.cdiv(M, BM), triton.cdiv(N, BN), BH)
        _qk_kernel[grid1](q, k, S, M, N, D, scale,
                          q.stride(0), q.stride(1), q.stride(2),
                          k.stride(0), k.stride(1), k.stride(2),
                          S.stride(0), S.stride(1), S.stride(2),
                          BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=3)

        BLOCK = triton.next_power_of_2(N)
        _softmax_kernel[(BH * M,)](S, P, N, BLOCK=BLOCK, num_warps=4)

        BD = 64
        grid3 = (triton.cdiv(M, BM), triton.cdiv(Dv, BD), BH)
        _pv_kernel[grid3](P, v, O, M, N, Dv,
                          P.stride(0), P.stride(1), P.stride(2),
                          v.stride(0), v.stride(1), v.stride(2),
                          O.stride(0), O.stride(1), O.stride(2),
                          BM=BM, BD=BD, BK=BK, num_warps=4, num_stages=3)

        return O.reshape(B, H, M, Dv)
