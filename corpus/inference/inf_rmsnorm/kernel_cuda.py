import torch
import torch.nn as nn

from smforge.cuda import build

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define THREADS 256

template <typename T>
__global__ void __launch_bounds__(THREADS) rmsnorm_kernel(const T* __restrict__ x,
                                                          const float* __restrict__ w,
                                                          float* __restrict__ y,
                                                          long rows, long H, float eps) {
  long row = blockIdx.x;
  if (row >= rows) return;
  const T* xr = x + row * H;
  float* yr = y + row * H;

  float s = 0.f;
  for (long i = threadIdx.x; i < H; i += THREADS) {
    float v = float(xr[i]);
    s += v * v;
  }
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);

  __shared__ float red[THREADS / 32];
  __shared__ float total;
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) red[warp] = s;
  __syncthreads();
  if (warp == 0) {
    float t = (lane < THREADS / 32) ? red[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffff, t, o);
    if (lane == 0) total = t;
  }
  __syncthreads();

  float rstd = rsqrtf(total / (float)H + eps);
  for (long i = threadIdx.x; i < H; i += THREADS) {
    float v = float(xr[i]) * rstd;
    float vr = float(T(v));   // round to input dtype like h.to(x.dtype)
    yr[i] = w[i] * vr;
  }
}

template <typename T>
static void launch(const void* x, const float* w, float* y, long rows, long H, float eps,
                   cudaStream_t stream) {
  if (rows <= 0) return;
  rmsnorm_kernel<T><<<(unsigned)rows, THREADS, 0, stream>>>((const T*)x, w, y, rows, H, eps);
}

void rmsnorm(int dtype, const void* x, const float* w, float* y, long rows, long H, float eps,
             cudaStream_t stream) {
  if (dtype == 1) launch<__half>(x, w, y, rows, H, eps, stream);
  else if (dtype == 2) launch<__nv_bfloat16>(x, w, y, rows, H, eps, stream);
  else launch<float>(x, w, y, rows, H, eps, stream);
}
"""

CPP_SRC = r"""
void rmsnorm(int dtype, const void* x, const float* w, float* y, long rows, long H, float eps,
             cudaStream_t stream);

torch::Tensor forward(torch::Tensor x, torch::Tensor w, double eps) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda());
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous());
  TORCH_CHECK(w.scalar_type() == torch::kFloat);
  auto type = x.scalar_type();
  TORCH_CHECK(type == torch::kHalf || type == torch::kBFloat16 || type == torch::kFloat);
  int dtype = type == torch::kHalf ? 1 : type == torch::kBFloat16 ? 2 : 0;
  long H = x.size(-1);
  TORCH_CHECK(w.numel() == H);
  long rows = H > 0 ? x.numel() / H : 0;
  auto y = torch::empty(x.sizes(), x.options().dtype(torch::kFloat));
  rmsnorm(dtype, x.data_ptr(), w.data_ptr<float>(), y.data_ptr<float>(), rows, H, (float)eps,
          at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
"""

ext = build(CUDA_SRC, CPP_SRC, ["forward"])


class Model(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(1 + 0.1 * torch.randn(hidden_size))
        self.eps = eps

    def forward(self, x):
        out_dtype = torch.promote_types(self.weight.dtype, x.dtype)
        w = self.weight.detach().to(torch.float32).contiguous()
        y = ext.forward(x.contiguous(), w, float(self.eps))
        if y.dtype != out_dtype:
            y = y.to(out_dtype)
        return y
