import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def forward(self, q, k_cache, v_cache):
        return F.scaled_dot_product_attention(q, k_cache, v_cache, enable_gqa=True)
