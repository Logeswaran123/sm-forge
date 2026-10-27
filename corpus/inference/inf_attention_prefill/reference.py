import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
