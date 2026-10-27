import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(1 + 0.1 * torch.randn(hidden_size))
        self.eps = eps

    def forward(self, x, residual):
        residual = x + residual
        h = residual.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(x.dtype), residual
