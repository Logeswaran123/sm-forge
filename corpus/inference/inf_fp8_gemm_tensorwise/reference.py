import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        weight = 0.05 * torch.randn(out_features, in_features)
        scale = weight.abs().amax() / 448.0
        self.register_buffer("weight", (weight / scale).to(torch.float8_e4m3fn))
        self.register_buffer("weight_scale", scale)
        self.register_buffer("input_scale", torch.tensor(0.05))

    def forward(self, x):
        out = x.float() @ self.weight.float().t()
        return (out * (self.input_scale * self.weight_scale)).to(torch.bfloat16)
