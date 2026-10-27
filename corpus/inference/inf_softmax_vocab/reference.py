import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, logits):
        return torch.softmax(logits.float(), dim=-1)
