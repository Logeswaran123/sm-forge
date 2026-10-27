import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _bn_eval_kernel(
    x_ptr, y_ptr,
    mean_ptr, var_ptr, w_ptr, b_ptr,
    C, HW, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)          # index over N*C
    blk = tl.program_id(1)          # block over HW
    c = row % C

    mean = tl.load(mean_ptr + c).to(tl.float32)
    var = tl.load(var_ptr + c).to(tl.float32)
    w = tl.load(w_ptr + c).to(tl.float32)
    b = tl.load(b_ptr + c).to(tl.float32)
    scale = w / tl.sqrt(var + eps)
    shift = b - mean * scale

    offs = blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < HW
    base = row.to(tl.int64) * HW
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * scale + shift
    tl.store(y_ptr + base + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


class Model(nn.Module):
    def __init__(self, num_features: int):
        super(Model, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C = x.shape[0], x.shape[1]
        x_c = x.contiguous()
        HW = x_c.numel() // (N * C) if N * C > 0 else 0
        y = torch.empty_like(x_c)
        if x_c.numel() == 0:
            return y

        mean = self.bn.running_mean.contiguous()
        var = self.bn.running_var.contiguous()
        w = self.bn.weight.contiguous()
        b = self.bn.bias.contiguous()

        BLOCK = 1024
        grid = (N * C, triton.cdiv(HW, BLOCK))
        _bn_eval_kernel[grid](
            x_c, y,
            mean, var, w, b,
            C, HW, float(self.bn.eps),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y.view(x.shape)
