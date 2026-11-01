import torch
import torch.nn as nn

from smforge.cuda import build

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// x viewed as [outer, red, inner]; y as [outer, inner]
template <typename T>
__global__ void __launch_bounds__(256) mean_dim_kernel(const T* __restrict__ x, T* __restrict__ y,
                                                       long outer, long red, long inner) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long total = outer * inner;
  if (idx >= total) return;
  long o = idx / inner;
  long i = idx - o * inner;
  const T* p = x + o * red * inner + i;
  float acc = 0.f;
  for (long r = 0; r < red; ++r) {
    acc += float(p[r * inner]);
  }
  y[idx] = T(acc / (float)red);
}

template <typename T>
static void launch(const void* x, void* y, long outer, long red, long inner, cudaStream_t stream) {
  long total = outer * inner;
  if (total <= 0) return;
  long blocks = (total + 255) / 256;
  mean_dim_kernel<T><<<(unsigned)blocks, 256, 0, stream>>>((const T*)x, (T*)y, outer, red, inner);
}

void mean_dim(int dtype, const void* x, void* y, long outer, long red, long inner,
              cudaStream_t stream) {
  if (dtype == 1) launch<__half>(x, y, outer, red, inner, stream);
  else if (dtype == 2) launch<__nv_bfloat16>(x, y, outer, red, inner, stream);
  else launch<float>(x, y, outer, red, inner, stream);
}
"""

CPP_SRC = r"""
void mean_dim(int dtype, const void* x, void* y, long outer, long red, long inner,
              cudaStream_t stream);

torch::Tensor forward(torch::Tensor x, int64_t dim) {
  TORCH_CHECK(x.is_cuda());
  x = x.contiguous();
  int64_t nd = x.dim();
  if (dim < 0) dim += nd;
  TORCH_CHECK(dim >= 0 && dim < nd);
  long outer = 1, inner = 1;
  for (int64_t d = 0; d < dim; ++d) outer *= x.size(d);
  for (int64_t d = dim + 1; d < nd; ++d) inner *= x.size(d);
  long red = x.size(dim);
  std::vector<int64_t> out_shape;
  for (int64_t d = 0; d < nd; ++d) if (d != dim) out_shape.push_back(x.size(d));
  auto y = torch::empty(out_shape, x.options());
  auto type = x.scalar_type();
  TORCH_CHECK(type == torch::kFloat || type == torch::kHalf || type == torch::kBFloat16);
  int dtype = type == torch::kHalf ? 1 : type == torch::kBFloat16 ? 2 : 0;
  mean_dim(dtype, x.data_ptr(), y.data_ptr(), outer, red, inner,
           at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
"""

ext = build(CUDA_SRC, CPP_SRC, ["forward"])


class Model(nn.Module):
    def __init__(self, dim: int):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ext.forward(x.contiguous(), int(self.dim))
