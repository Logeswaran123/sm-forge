import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Parameter(0.02 * torch.randn(num_experts, hidden_size))
        self.w13 = nn.Parameter(0.02 * torch.randn(num_experts, 2 * intermediate_size, hidden_size))
        self.w2 = nn.Parameter(0.02 * torch.randn(num_experts, hidden_size, intermediate_size))

    def forward(self, x):
        probs = torch.softmax(x.float() @ self.gate.float().t(), dim=-1)
        weights, experts = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        for expert in range(self.w13.shape[0]):
            token, slot = torch.where(experts == expert)
            if token.numel() == 0:
                continue
            gate, up = (x[token] @ self.w13[expert].t()).chunk(2, dim=-1)
            y = (F.silu(gate) * up) @ self.w2[expert].t()
            out.index_add_(0, token, y.float() * weights[token, slot, None])
        return out.to(x.dtype)
