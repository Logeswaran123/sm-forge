import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _softmax_row_kernel(
    x_ptr, y_ptr,
    N,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    # Fallback two-pass kernel (used only when the split scheme does not apply).
    row = tl.program_id(0)
    x_row = x_ptr + row.to(tl.int64) * stride_x
    y_row = y_ptr + row.to(tl.int64) * stride_y
    offs = tl.arange(0, BLOCK)
    m = tl.full([BLOCK], float("-inf"), dtype=tl.float32)
    s = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, N, BLOCK):
        cols = start + offs
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, x)
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        s = s * tl.exp(m - m_safe) + tl.exp(x - m_safe)
        m = m_new
    m_row = tl.max(m, axis=0)
    m_row_safe = tl.where(m_row == float("-inf"), 0.0, m_row)
    m_lane_safe = tl.where(m == float("-inf"), m_row_safe, m)
    s_row = tl.sum(s * tl.exp(m_lane_safe - m_row_safe), axis=0)
    inv = 1.0 / s_row
    for start in range(0, N, BLOCK):
        cols = start + offs
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=float("-inf")).to(tl.float32)
        y = tl.exp(x - m_row_safe) * inv
        tl.store(y_row + cols, y, mask=mask)


@triton.jit
def _zero_i32_kernel(p, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(p + offs, tl.zeros([BLOCK], dtype=tl.int32), mask=offs < n)


@triton.jit
def _softmax_split_kernel(
    x_ptr, y_ptr, pm_ptr, ps_ptr, cnt_ptr,
    M, N, stride_x, stride_y,
    G, CHUNK,
    BLOCK: tl.constexpr, GP: tl.constexpr,
):
    pid = tl.program_id(0)
    P = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    goffs = tl.arange(0, GP)
    gmask = goffs < G
    total = M * G
    for t in range(pid, total, P):
        row = t // G
        c = t - row * G
        cols = c * CHUNK + offs
        mask = (offs < CHUNK) & (cols < N)
        x = tl.load(x_ptr + row.to(tl.int64) * stride_x + cols,
                    mask=mask, other=float("-inf")).to(tl.float32)
        m_loc = tl.max(x, axis=0)
        m_loc_safe = tl.where(m_loc == float("-inf"), 0.0, m_loc)
        e = tl.exp(x - m_loc_safe)
        s_loc = tl.sum(e, axis=0)

        # publish partials
        tl.store(pm_ptr + t, m_loc)
        tl.store(ps_ptr + t, s_loc)
        tl.debug_barrier()
        cnt = tl.atomic_add(cnt_ptr + row, 1, sem="acq_rel")
        while cnt < G:
            cnt = tl.atomic_add(cnt_ptr + row, 0, sem="acquire")
        tl.debug_barrier()

        # combine partials
        pms = tl.load(pm_ptr + row * G + goffs, mask=gmask,
                      other=float("-inf"), volatile=True)
        pss = tl.load(ps_ptr + row * G + goffs, mask=gmask,
                      other=0.0, volatile=True)
        m_row = tl.max(pms, axis=0)
        m_row_safe = tl.where(m_row == float("-inf"), 0.0, m_row)
        s_row = tl.sum(pss * tl.exp(pms - m_row_safe), axis=0)
        f = tl.exp(m_loc_safe - m_row_safe) / s_row

        # depart; last one resets counter for the next call
        tl.debug_barrier()
        old = tl.atomic_add(cnt_ptr + row, 1, sem="acq_rel")
        tl.atomic_xchg(cnt_ptr + row, 0, mask=old == 2 * G - 1)

        y = e * f
        tl.store(y_ptr + row.to(tl.int64) * stride_y + cols, y, mask=mask)


_SM_COUNT = {}
_OCC = {}

_BLOCK_MAX = 16384
_NUM_WARPS = 8


def _sm_count(dev):
    key = dev.index if dev.index is not None else torch.cuda.current_device()
    if key not in _SM_COUNT:
        _SM_COUNT[key] = torch.cuda.get_device_properties(key).multi_processor_count
    return _SM_COUNT[key]


def _occupancy(args, BLOCK, GP):
    key = (BLOCK, GP)
    if key in _OCC:
        return _OCC[key]
    occ = 1
    try:
        ck = _softmax_split_kernel.warmup(*args, BLOCK=BLOCK, GP=GP,
                                          num_warps=_NUM_WARPS, grid=(1,))
        if hasattr(ck, "_init_handles"):
            ck._init_handles()
        regs = int(ck.n_regs)
        smem = int(ck.metadata.shared)
        threads = 32 * _NUM_WARPS
        regs_alloc = max(8, ((regs + 7) // 8) * 8)
        by_regs = 65536 // (regs_alloc * threads)
        by_threads = 1536 // threads
        by_smem = (100 * 1024) // max(smem, 1) if smem > 0 else 64
        occ = max(1, min(by_regs, by_threads, by_smem, 4))
    except Exception:
        occ = 1
    _OCC[key] = occ
    return occ


class Model(nn.Module):
    def _counters(self, M, dev):
        buf = getattr(self, "_cnt_buf", None)
        if buf is None or buf.numel() < M or buf.device != dev:
            n = max(M, 256)
            buf = torch.empty(n, dtype=torch.int32, device=dev)
            _zero_i32_kernel[(triton.cdiv(n, 1024),)](buf, n, BLOCK=1024)
            self._cnt_buf = buf
        return buf

    def forward(self, logits):
        orig_shape = logits.shape
        N = orig_shape[-1]
        x = logits.reshape(-1, N)
        if x.stride(-1) != 1:
            x = x.contiguous()
        M = x.shape[0]
        dev = logits.device
        y = torch.empty((M, N), dtype=torch.float32, device=dev)
        if M == 0 or N == 0:
            return y.reshape(orig_shape)

        G = triton.cdiv(N, _BLOCK_MAX)
        CHUNK = triton.cdiv(N, G)
        BLOCK = triton.next_power_of_2(CHUNK)
        GP = max(2, triton.next_power_of_2(G))
        sms = _sm_count(dev)

        if G > sms:
            _softmax_row_kernel[(M,)](
                x, y, N, x.stride(0), y.stride(0),
                BLOCK=4096, num_warps=8,
            )
            return y.reshape(orig_shape)

        cnt = self._counters(M, dev)
        pm = torch.empty(M * G, dtype=torch.float32, device=dev)
        ps = torch.empty(M * G, dtype=torch.float32, device=dev)
        args = (x, y, pm, ps, cnt, M, N, x.stride(0), y.stride(0), G, CHUNK)
        occ = _occupancy(args, BLOCK, GP)
        P = sms * occ
        P = min(P, M * G)
        if P < G:
            P = G
        _softmax_split_kernel[(P,)](
            *args, BLOCK=BLOCK, GP=GP, num_warps=_NUM_WARPS,
        )
        return y.reshape(orig_shape)
