import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ce_row_kernel(
    x_ptr, t_ptr, loss_ptr, w_ptr,
    C, stride_row,
    IGNORE_INDEX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    row_ptr = x_ptr + row.to(tl.int64) * stride_row
    offs = tl.arange(0, BLOCK)

    m = tl.full([BLOCK], float("-inf"), tl.float32)
    s = tl.zeros([BLOCK], tl.float32)
    for start in range(0, C, BLOCK):
        cols = start + offs
        mask = cols < C
        x = tl.load(row_ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, x)
        # guard lanes where everything seen so far is -inf
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        s = s * tl.exp(m - m_safe) + tl.exp(x - m_safe)
        m = m_new

    m_row = tl.max(m, axis=0)
    m_row_safe = tl.where(m_row == float("-inf"), 0.0, m_row)
    m_lane_safe = tl.where(m == float("-inf"), m_row_safe, m)
    s_row = tl.sum(s * tl.exp(m_lane_safe - m_row_safe), axis=0)
    lse = m_row_safe + tl.log(s_row)

    t = tl.load(t_ptr + row)
    valid = t != IGNORE_INDEX
    t_safe = tl.where(valid, t, 0)
    xt = tl.load(row_ptr + t_safe).to(tl.float32)
    loss = tl.where(valid, lse - xt, 0.0)
    w = tl.where(valid, 1.0, 0.0)
    tl.store(loss_ptr + row, loss)
    tl.store(w_ptr + row, w)


@triton.jit
def _mean_kernel(loss_ptr, w_ptr, out_ptr, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], tl.float32)
    cnt = tl.zeros([BLOCK], tl.float32)
    for start in range(0, N, BLOCK):
        idx = start + offs
        mask = idx < N
        acc += tl.load(loss_ptr + idx, mask=mask, other=0.0)
        cnt += tl.load(w_ptr + idx, mask=mask, other=0.0)
    total = tl.sum(acc, axis=0)
    count = tl.sum(cnt, axis=0)
    res = total / count
    tl.store(out_ptr, res.to(out_ptr.dtype.element_ty))


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        x = predictions.contiguous()
        t = targets.contiguous()
        N, C = x.shape
        loss = torch.empty((N,), device=x.device, dtype=torch.float32)
        w = torch.empty((N,), device=x.device, dtype=torch.float32)
        out = torch.empty((), device=x.device, dtype=x.dtype)

        BLOCK = min(2048, triton.next_power_of_2(C))
        BLOCK = max(BLOCK, 128)
        _ce_row_kernel[(N,)](
            x, t, loss, w,
            C, x.stride(0),
            IGNORE_INDEX=-100,
            BLOCK=BLOCK,
            num_warps=8,
        )
        _mean_kernel[(1,)](loss, w, out, N, BLOCK=1024, num_warps=4)
        return out
