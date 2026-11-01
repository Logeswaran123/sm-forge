import torch
import torch.nn as nn

from smforge.cuda import build

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define BM 64
#define BN 64
#define BK 16

template <typename T>
__global__ void __launch_bounds__(256) gemm_kernel(const T* __restrict__ A, const T* __restrict__ B,
                                                   T* __restrict__ C, int M, int N, int K) {
  __shared__ float As[BM][BK + 1];
  __shared__ float Bs[BK][BN];

  int tid = threadIdx.x;
  int tx = tid % 16;
  int ty = tid / 16;
  int rowBase = blockIdx.y * BM;
  int colBase = blockIdx.x * BN;

  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

  for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      int e = tid + 256 * r;
      int ar = e / BK, ac = e % BK;
      int gr = rowBase + ar, gc = k0 + ac;
      As[ar][ac] = (gr < M && gc < K) ? float(A[(long)gr * K + gc]) : 0.f;
      int br = e / BN, bc = e % BN;
      int gbr = k0 + br, gbc = colBase + bc;
      Bs[br][bc] = (gbr < K && gbc < N) ? float(B[(long)gbr * N + gbc]) : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      float a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) a[i] = As[ty + 16 * i][k];
#pragma unroll
      for (int j = 0; j < 4; ++j) b[j] = Bs[k][tx + 16 * j];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int r = rowBase + ty + 16 * i;
    if (r >= M) continue;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int c = colBase + tx + 16 * j;
      if (c < N) C[(long)r * N + c] = T(acc[i][j]);
    }
  }
}

template <typename T>
static void launch(const void* A, const void* B, void* C, int M, int N, int K, cudaStream_t stream) {
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  gemm_kernel<T><<<grid, 256, 0, stream>>>((const T*)A, (const T*)B, (T*)C, M, N, K);
}

void gemm(int dtype, const void* A, const void* B, void* C, int M, int N, int K, cudaStream_t stream) {
  if (dtype == 1) launch<__half>(A, B, C, M, N, K, stream);
  else if (dtype == 2) launch<__nv_bfloat16>(A, B, C, M, N, K, stream);
  else launch<float>(A, B, C, M, N, K, stream);
}
"""

CPP_SRC = r"""
void gemm(int dtype, const void* A, const void* B, void* C, int M, int N, int K, cudaStream_t stream);

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda());
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2);
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous());
  TORCH_CHECK(A.scalar_type() == B.scalar_type());
  TORCH_CHECK(A.size(1) == B.size(0));
  int M = A.size(0), K = A.size(1), N = B.size(1);
  auto type = A.scalar_type();
  TORCH_CHECK(type == torch::kFloat || type == torch::kHalf || type == torch::kBFloat16);
  int dtype = type == torch::kHalf ? 1 : type == torch::kBFloat16 ? 2 : 0;
  auto C = torch::empty({M, N}, A.options());
  if (M == 0 || N == 0) return C;
  gemm(dtype, A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N, K, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return C;
}
"""

ext = build(CUDA_SRC, CPP_SRC, ["forward"])


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return ext.forward(A.contiguous(), B.contiguous())
