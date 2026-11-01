import torch
import torch.nn as nn

from smforge.cuda import build

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void __launch_bounds__(256) swiglu_kernel(const T* __restrict__ x, T* __restrict__ y,
                                                     long rows, long H) {
  long total = rows * H;
  long stride = (long)gridDim.x * blockDim.x;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total; i += stride) {
    long r = i / H;
    long c = i - r * H;
    const T* xr = x + r * 2 * H;
    float g = float(xr[c]);
    float u = float(xr[H + c]);
    float s = g / (1.0f + expf(-g));
    y[i] = T(s * u);
  }
}

template <typename T>
static void launch(const void* x, void* y, long rows, long H, cudaStream_t stream) {
  long total = rows * H;
  if (total <= 0) return;
  long blocks = (total + 255) / 256;
  if (blocks > 84L * 64) blocks = 84L * 64;
  swiglu_kernel<T><<<(unsigned)blocks, 256, 0, stream>>>((const T*)x, (T*)y, rows, H);
}

void swiglu(int dtype, const void* x, void* y, long rows, long H, cudaStream_t stream) {
  if (dtype == 1) launch<__half>(x, y, rows, H, stream);
  else if (dtype == 2) launch<__nv_bfloat16>(x, y, rows, H, stream);
  else launch<float>(x, y, rows, H, stream);
}
"""

CPP_SRC = r"""
void swiglu(int dtype, const void* x, void* y, long rows, long H, cudaStream_t stream);

torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda());
  TORCH_CHECK(x.dim() >= 1);
  x = x.contiguous();
  long D = x.size(-1);
  TORCH_CHECK(D % 2 == 0, "last dim must be even");
  long H = D / 2;
  long rows = D > 0 ? x.numel() / D : 0;
  auto sizes = x.sizes().vec();
  sizes.back() = H;
  auto y = torch::empty(sizes, x.options());
  auto type = x.scalar_type();
  int dtype = type == torch::kHalf ? 1 : type == torch::kBFloat16 ? 2 : 0;
  TORCH_CHECK(dtype != 0 || type == torch::kFloat, "unsupported dtype");
  swiglu(dtype, x.data_ptr(), y.data_ptr(), rows, H, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
"""

ext = build(CUDA_SRC, CPP_SRC, ["forward"])


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ext.forward(x)
