import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.relu(self.linear(x))


def get_inputs():
    return [torch.randn(256, 128)]


def get_init_inputs():
    return [128, 192]
