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
    m = tl.full([BLOCK], float("-inf"), dtype=tl.float32)
    for start in range(0, N, BLOCK):
        x = tl.load(x_row + start + offs, mask=start + offs < N, other=float("-inf"))
        m = tl.maximum(m, x.to(tl.float32))
    m_row = tl.max(m, axis=0)
    s = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, N, BLOCK):
        x = tl.load(x_row + start + offs, mask=start + offs < N, other=float("-inf"))
        s += tl.exp(x.to(tl.float32) - m_row)
    inv = 1.0 / tl.sum(s, axis=0)
    for start in range(0, N, BLOCK):
        mask = start + offs < N
        x = tl.load(x_row + start + offs, mask=mask, other=float("-inf"))
        tl.store(y_row + start + offs, tl.exp(x.to(tl.float32) - m_row) * inv, mask=mask)


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
