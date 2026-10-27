import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def router_kernel(x_ptr, g_ptr, w_ptr, idx_ptr, H, E,
                  K: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_H: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    offs_e = tl.arange(0, BLOCK_E)
    me = offs_e < E
    acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    for h0 in range(0, H, BLOCK_H):
        oh = h0 + tl.arange(0, BLOCK_H)
        mh = oh < H
        xv = tl.load(x_ptr + t * H + oh, mask=mh, other=0.0).to(tl.float32)
        gv = tl.load(g_ptr + offs_e[:, None] * H + oh[None, :],
                     mask=me[:, None] & mh[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(gv * xv[None, :], axis=1)
    logits = tl.where(me, acc, float("-inf"))
    mx = tl.max(logits, axis=0)
    p = tl.exp(logits - mx)
    p = tl.where(me, p, 0.0)
    p = p / tl.sum(p, axis=0)

    # pass 1: sum of top-k
    work = tl.where(me, p, -1.0)
    wsum = 0.0
    for k in tl.static_range(K):
        m = tl.max(work, axis=0)
        i = tl.min(tl.where(work == m, offs_e, BLOCK_E), axis=0)
        wsum += m
        work = tl.where(offs_e == i, -1.0, work)
    # pass 2: store normalized
    work = tl.where(me, p, -1.0)
    for k in tl.static_range(K):
        m = tl.max(work, axis=0)
        i = tl.min(tl.where(work == m, offs_e, BLOCK_E), axis=0)
        tl.store(w_ptr + t * K + k, m / wsum)
        tl.store(idx_ptr + t * K + k, i.to(tl.int32))
        work = tl.where(offs_e == i, -1.0, work)


@triton.jit
def bucket_kernel(idx_ptr, sorted_ptr, cnt_ptr, TK, T, BLOCK: tl.constexpr):
    e = tl.program_id(0)
    base = tl.zeros([], dtype=tl.int32)
    for s in range(0, TK, BLOCK):
        offs = s + tl.arange(0, BLOCK)
        ex = tl.load(idx_ptr + offs, mask=offs < TK, other=-1)
        m = ex == e
        mi = m.to(tl.int32)
        pos = tl.cumsum(mi, axis=0) - mi + base
        tl.store(sorted_ptr + e.to(tl.int64) * T + pos, offs.to(tl.int32), mask=m)
        base += tl.sum(mi, axis=0)
    tl.store(cnt_ptr + e, base)


@triton.jit
def gemm1_kernel(x_ptr, w13_ptr, sorted_ptr, cnt_ptr, h_ptr, T, H, I,
                 K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    e = tl.program_id(2).to(tl.int64)
    cnt = tl.load(cnt_ptr + e)
    if pid_m * BM >= cnt:
        return
    offs_m = pid_m * BM + tl.arange(0, BM)
    mm = offs_m < cnt
    pair = tl.load(sorted_ptr + e * T + offs_m, mask=mm, other=0).to(tl.int64)
    tok = pair // K
    offs_n = pid_n * BN + tl.arange(0, BN)
    mn = offs_n < I
    wbase = w13_ptr + e * (2 * I) * H
    acc_g = tl.zeros([BM, BN], dtype=tl.float32)
    acc_u = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, H, BK):
        offs_k = k0 + tl.arange(0, BK)
        mk = offs_k < H
        a = tl.load(x_ptr + tok[:, None] * H + offs_k[None, :],
                    mask=mm[:, None] & mk[None, :], other=0.0)
        bmask = mk[:, None] & mn[None, :]
        bg = tl.load(wbase + offs_n[None, :].to(tl.int64) * H + offs_k[:, None],
                     mask=bmask, other=0.0).to(a.dtype)
        bu = tl.load(wbase + (offs_n[None, :].to(tl.int64) + I) * H + offs_k[:, None],
                     mask=bmask, other=0.0).to(a.dtype)
        acc_g = tl.dot(a, bg, acc_g)
        acc_u = tl.dot(a, bu, acc_u)
    sg = acc_g / (1.0 + tl.exp(-acc_g))
    h = sg * acc_u
    tl.store(h_ptr + pair[:, None] * I + offs_n[None, :],
             h.to(h_ptr.dtype.element_ty), mask=mm[:, None] & mn[None, :])


@triton.jit
def gemm2_kernel(h_ptr, w2_ptr, sorted_ptr, cnt_ptr, wt_ptr, y_ptr, T, H, I,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    e = tl.program_id(2).to(tl.int64)
    cnt = tl.load(cnt_ptr + e)
    if pid_m * BM >= cnt:
        return
    offs_m = pid_m * BM + tl.arange(0, BM)
    mm = offs_m < cnt
    pair = tl.load(sorted_ptr + e * T + offs_m, mask=mm, other=0).to(tl.int64)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mn = offs_n < H
    wbase = w2_ptr + e * H * I
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, I, BK):
        offs_k = k0 + tl.arange(0, BK)
        mk = offs_k < I
        a = tl.load(h_ptr + pair[:, None] * I + offs_k[None, :],
                    mask=mm[:, None] & mk[None, :], other=0.0)
        b = tl.load(wbase + offs_n[None, :].to(tl.int64) * I + offs_k[:, None],
                    mask=mk[:, None] & mn[None, :], other=0.0).to(a.dtype)
        acc = tl.dot(a, b, acc)
    w = tl.load(wt_ptr + pair, mask=mm, other=0.0)
    acc = acc * w[:, None]
    tl.store(y_ptr + pair[:, None] * H + offs_n[None, :], acc,
             mask=mm[:, None] & mn[None, :])


@triton.jit
def combine_kernel(y_ptr, out_ptr, H, K: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = offs < H
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(K):
        acc += tl.load(y_ptr + (t * K + k) * H + offs, mask=m, other=0.0)
    tl.store(out_ptr + t * H + offs, acc.to(out_ptr.dtype.element_ty), mask=m)


class Model(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Parameter(0.02 * torch.randn(num_experts, hidden_size))
        self.w13 = nn.Parameter(0.02 * torch.randn(num_experts, 2 * intermediate_size, hidden_size))
        self.w2 = nn.Parameter(0.02 * torch.randn(num_experts, hidden_size, intermediate_size))

    def forward(self, x):
        orig_shape = x.shape
        H = orig_shape[-1]
        x2 = x.reshape(-1, H).contiguous()
        T = x2.shape[0]
        E = self.gate.shape[0]
        I = self.w13.shape[1] // 2
        K = self.top_k
        dev = x.device

        gate = self.gate.contiguous()
        w13 = self.w13.contiguous()
        w2 = self.w2.contiguous()

        weights = torch.empty((T * K,), dtype=torch.float32, device=dev)
        experts = torch.empty((T * K,), dtype=torch.int32, device=dev)
        BLOCK_E = triton.next_power_of_2(E)
        router_kernel[(T,)](x2, gate, weights, experts, H, E,
                            K=K, BLOCK_E=BLOCK_E, BLOCK_H=64, num_warps=4)

        sorted_ids = torch.empty((E * T,), dtype=torch.int32, device=dev)
        counts = torch.empty((E,), dtype=torch.int32, device=dev)
        bucket_kernel[(E,)](experts, sorted_ids, counts, T * K, T, BLOCK=1024, num_warps=4)

        hbuf = torch.empty((T * K, I), dtype=x.dtype, device=dev)
        BM, BN, BK = 64, 64, 64
        grid1 = (triton.cdiv(I, BN), triton.cdiv(T, BM), E)
        gemm1_kernel[grid1](x2, w13, sorted_ids, counts, hbuf, T, H, I,
                            K=K, BM=BM, BN=BN, BK=BK, num_warps=4)

        ybuf = torch.empty((T * K, H), dtype=torch.float32, device=dev)
        grid2 = (triton.cdiv(H, BN), triton.cdiv(T, BM), E)
        gemm2_kernel[grid2](hbuf, w2, sorted_ids, counts, weights, ybuf, T, H, I,
                            BM=BM, BN=BN, BK=BK, num_warps=4)

        out = torch.empty((T, H), dtype=x.dtype, device=dev)
        CB = 1024
        combine_kernel[(T, triton.cdiv(H, CB))](ybuf, out, H, K=K, BLOCK=CB, num_warps=4)
        return out.reshape(orig_shape)
