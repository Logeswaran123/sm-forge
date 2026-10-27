import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, head_dim, max_position, base):
        super().__init__()
        inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        freqs = torch.outer(torch.arange(max_position, dtype=torch.float32), inv_freq)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, positions, q, k):
        cos = self.cos[positions].float().unsqueeze(1)
        sin = self.sin[positions].float().unsqueeze(1)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    @staticmethod
    def _rotate(x, cos, sin):
        x1, x2 = x.float().chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).to(x.dtype)
