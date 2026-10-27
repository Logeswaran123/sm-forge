import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def forward(self, x):
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up
