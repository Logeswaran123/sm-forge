import torch
import torch.nn as nn

from smforge.cuda import build

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

constexpr int TILE = 16;

template <typename T>
__global__ void __launch_bounds__(TILE * TILE) linear_relu_kernel(
    const T* x, const T* w, const T* b, T* y, int M, int N, int K) {
  __shared__ float xs[TILE][TILE + 1];
  __shared__ float ws[TILE][TILE + 1];
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  int w_row = blockIdx.x * TILE + threadIdx.y;
  float acc = 0.0f;
  for (int k0 = 0; k0 < K; k0 += TILE) {
    int k = k0 + threadIdx.x;
    xs[threadIdx.y][threadIdx.x] = row < M && k < K ? float(x[(long)row * K + k]) : 0.0f;
    ws[threadIdx.x][threadIdx.y] = w_row < N && k < K ? float(w[(long)w_row * K + k]) : 0.0f;
    __syncthreads();
    for (int i = 0; i < TILE; ++i) acc += xs[threadIdx.y][i] * ws[i][threadIdx.x];
    __syncthreads();
  }
  if (row < M && col < N) y[(long)row * N + col] = T(fmaxf(acc + float(b[col]), 0.0f));
}

template <typename T>
static void launch(const void* x, const void* w, const void* b, void* y, int M, int N, int K,
                   cudaStream_t stream) {
  dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  linear_relu_kernel<T><<<grid, block, 0, stream>>>(
      (const T*)x, (const T*)w, (const T*)b, (T*)y, M, N, K);
}

void linear_relu(int dtype, const void* x, const void* w, const void* b, void* y, int M, int N,
                 int K, cudaStream_t stream) {
  if (dtype == 1) launch<__half>(x, w, b, y, M, N, K, stream);
  else if (dtype == 2) launch<__nv_bfloat16>(x, w, b, y, M, N, K, stream);
  else launch<float>(x, w, b, y, M, N, K, stream);
}
"""

CPP_SRC = r"""
void linear_relu(int dtype, const void* x, const void* w, const void* b, void* y, int M, int N,
                 int K, cudaStream_t stream);

torch::Tensor forward(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && b.is_contiguous());
  auto type = x.scalar_type();
  TORCH_CHECK(type == torch::kFloat || type == torch::kHalf || type == torch::kBFloat16);
  int dtype = type == torch::kHalf ? 1 : type == torch::kBFloat16 ? 2 : 0;
  auto y = torch::empty({x.size(0), w.size(0)}, x.options());
  linear_relu(dtype, x.data_ptr(), w.data_ptr(), b.data_ptr(), y.data_ptr(), x.size(0),
              w.size(0), x.size(1), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
"""

ext = build(CUDA_SRC, CPP_SRC, ["forward"])


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        bound = 1 / in_features**0.5
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return ext.forward(x.contiguous(), self.weight, self.bias)


def get_inputs():
    return [torch.randn(256, 128)]


def get_init_inputs():
    return [128, 192]
