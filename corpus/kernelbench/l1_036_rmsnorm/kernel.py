import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rmsnorm_dim1_kernel(
    x_ptr, y_ptr,
    C, S, eps,
    BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_c = tl.arange(0, BLOCK_C)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_c = offs_c < C
    mask_s = offs_s < S
    mask = mask_c[:, None] & mask_s[None, :]

    base = pid_b.to(tl.int64) * C * S
    offs = base + offs_c[:, None].to(tl.int64) * S + offs_s[None, :]

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    ss = tl.sum(x * x, axis=0)          # [BLOCK_S]
    mean = ss / C
    inv = 1.0 / tl.sqrt(mean + eps)
    y = x * inv[None, :]
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


class Model(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        B = x.shape[0]
        C = x.shape[1]
        S = 1
        for d in x.shape[2:]:
            S *= d
        y = torch.empty_like(x)
        if x.numel() == 0:
            return y
        BLOCK_C = triton.next_power_of_2(C)
        BLOCK_S = max(16, min(128, 8192 // BLOCK_C))
        BLOCK_S = min(BLOCK_S, triton.next_power_of_2(S))
        grid = (triton.cdiv(S, BLOCK_S), B)
        rmsnorm_dim1_kernel[grid](
            x, y, C, S, float(self.eps),
            BLOCK_C=BLOCK_C, BLOCK_S=BLOCK_S,
            num_warps=8,
        )
        return y
