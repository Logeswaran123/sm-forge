import torch
import torch.nn as nn

from smforge.cuda import build

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void __launch_bounds__(256) rope_kernel(const long long* __restrict__ pos,
                                                   const float* __restrict__ cosb,
                                                   const float* __restrict__ sinb,
                                                   const T* __restrict__ x, T* __restrict__ y,
                                                   long ntok, long heads, long half, long maxpos) {
  long total = ntok * heads * half;
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= total) return;
  long i = idx % half;
  long rest = idx / half;       // token*heads + h
  long tok = rest / heads;
  long p = pos[tok];
  if (p < 0) p += maxpos;
  if (p < 0 || p >= maxpos) p = 0;
  float c = cosb[p * half + i];
  float s = sinb[p * half + i];
  long base = rest * (2 * half);
  float x1 = float(x[base + i]);
  float x2 = float(x[base + i + half]);
  y[base + i] = T(x1 * c - x2 * s);
  y[base + i + half] = T(x2 * c + x1 * s);
}

template <typename T>
static void launch(const long long* pos, const float* c, const float* s, const void* x, void* y,
                   long ntok, long heads, long half, long maxpos, cudaStream_t stream) {
  long total = ntok * heads * half;
  if (total <= 0) return;
  long blocks = (total + 255) / 256;
  rope_kernel<T><<<(unsigned)blocks, 256, 0, stream>>>(pos, c, s, (const T*)x, (T*)y,
                                                      ntok, heads, half, maxpos);
}

void rope(int dtype, const long long* pos, const float* c, const float* s, const void* x, void* y,
          long ntok, long heads, long half, long maxpos, cudaStream_t stream) {
  if (dtype == 1) launch<__half>(pos, c, s, x, y, ntok, heads, half, maxpos, stream);
  else if (dtype == 2) launch<__nv_bfloat16>(pos, c, s, x, y, ntok, heads, half, maxpos, stream);
  else launch<float>(pos, c, s, x, y, ntok, heads, half, maxpos, stream);
}
"""

CPP_SRC = r"""
void rope(int dtype, const long long* pos, const float* c, const float* s, const void* x, void* y,
          long ntok, long heads, long half, long maxpos, cudaStream_t stream);

static torch::Tensor rope_one(torch::Tensor positions, torch::Tensor x,
                              torch::Tensor cosb, torch::Tensor sinb) {
  TORCH_CHECK(x.is_contiguous());
  long half = cosb.size(1);
  long maxpos = cosb.size(0);
  long D = x.size(-1);
  TORCH_CHECK(D == 2 * half, "head_dim mismatch");
  long ntok = positions.numel();
  auto y = torch::empty_like(x);
  if (ntok == 0 || x.numel() == 0) return y;
  long heads = x.numel() / (ntok * D);
  TORCH_CHECK(heads * ntok * D == x.numel(), "shape mismatch");
  auto type = x.scalar_type();
  int dtype = type == torch::kHalf ? 1 : type == torch::kBFloat16 ? 2 : 0;
  TORCH_CHECK(dtype != 0 || type == torch::kFloat, "unsupported dtype");
  rope(dtype, (const long long*)positions.data_ptr(), (const float*)cosb.data_ptr(),
       (const float*)sinb.data_ptr(), x.data_ptr(), y.data_ptr(), ntok, heads, half, maxpos,
       at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

std::vector<torch::Tensor> forward(torch::Tensor positions, torch::Tensor q, torch::Tensor k,
                                   torch::Tensor cosb, torch::Tensor sinb) {
  TORCH_CHECK(positions.scalar_type() == torch::kLong && positions.is_contiguous());
  TORCH_CHECK(cosb.scalar_type() == torch::kFloat && cosb.is_contiguous());
  TORCH_CHECK(sinb.scalar_type() == torch::kFloat && sinb.is_contiguous());
  auto qo = rope_one(positions, q, cosb, sinb);
  auto ko = rope_one(positions, k, cosb, sinb);
  return {qo, ko};
}
"""

ext = build(CUDA_SRC, CPP_SRC, ["forward"])


class Model(nn.Module):
    def __init__(self, head_dim, max_position, base):
        super().__init__()
        inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        freqs = torch.outer(torch.arange(max_position, dtype=torch.float32), inv_freq)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, positions, q, k):
        pos = positions.to(torch.int64).contiguous()
        cosb = self.cos.to(torch.float32).contiguous()
        sinb = self.sin.to(torch.float32).contiguous()
        qo, ko = ext.forward(pos, q.contiguous(), k.contiguous(), cosb, sinb)
        return qo, ko
