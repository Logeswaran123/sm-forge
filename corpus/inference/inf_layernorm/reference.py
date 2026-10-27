import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(1 + 0.1 * torch.randn(hidden_size))
        self.bias = nn.Parameter(0.1 * torch.randn(hidden_size))
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
