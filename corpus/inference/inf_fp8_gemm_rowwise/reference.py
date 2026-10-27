import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        weight = 0.05 * torch.randn(out_features, in_features)
        scale = weight.abs().amax(dim=1) / 448.0
        self.register_buffer("weight", (weight / scale[:, None]).to(torch.float8_e4m3fn))
        self.register_buffer("weight_scale", scale)

    def forward(self, x, x_scale):
        out = x.float() @ self.weight.float().t()
        return (out * x_scale[:, None] * self.weight_scale[None, :]).to(torch.bfloat16)
