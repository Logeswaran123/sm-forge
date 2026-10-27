import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, q, key_cache, value_cache, block_tables, seq_lens):
        batch, heads, dim = q.shape
        keys = key_cache[block_tables.long()].flatten(1, 2).float()
        values = value_cache[block_tables.long()].flatten(1, 2).float()
        query = q.float().view(batch, keys.shape[2], heads // keys.shape[2], dim)
        scores = torch.einsum("bhgd,blhd->bhgl", query, keys) / dim**0.5
        positions = torch.arange(keys.shape[1], device=q.device)
        padding = positions[None, :] >= seq_lens[:, None].long()
        scores = scores.masked_fill(padding[:, None, None, :], float("-inf"))
        out = torch.einsum("bhgl,blhd->bhgd", scores.softmax(dim=-1), values)
        return out.reshape(batch, heads, dim).to(q.dtype)
