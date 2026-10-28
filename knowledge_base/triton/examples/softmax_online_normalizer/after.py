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
    row = tl.program_id(0)
    x_row = x_ptr + row.to(tl.int64) * stride_x
    y_row = y_ptr + row.to(tl.int64) * stride_y

    offs = tl.arange(0, BLOCK)

    # Pass 1: online per-lane max and sum
    m = tl.full([BLOCK], float("-inf"), dtype=tl.float32)
    s = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, N, BLOCK):
        cols = start + offs
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, x)
        # guard against (-inf) - (-inf) = nan for lanes not yet seen
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        s = s * tl.exp(m - m_safe) + tl.exp(x - m_safe)
        m = m_new

    m_row = tl.max(m, axis=0)
    m_row_safe = tl.where(m_row == float("-inf"), 0.0, m_row)
    m_lane_safe = tl.where(m == float("-inf"), m_row_safe, m)
    s_row = tl.sum(s * tl.exp(m_lane_safe - m_row_safe), axis=0)
    inv = 1.0 / s_row

    # Pass 2: write normalized output
    for start in range(0, N, BLOCK):
        cols = start + offs
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=float("-inf")).to(tl.float32)
        y = tl.exp(x - m_row_safe) * inv
        tl.store(y_row + cols, y, mask=mask)


class Model(nn.Module):
    def forward(self, logits):
        orig_shape = logits.shape
        N = orig_shape[-1]
        x = logits.reshape(-1, N)
        if x.stride(-1) != 1:
            x = x.contiguous()
        M = x.shape[0]
        y = torch.empty((M, N), dtype=torch.float32, device=logits.device)
        if M > 0 and N > 0:
            BLOCK = 4096
            _softmax_row_kernel[(M,)](
                x, y, N, x.stride(0), y.stride(0),
                BLOCK=BLOCK, num_warps=8,
            )
        return y.reshape(orig_shape)
