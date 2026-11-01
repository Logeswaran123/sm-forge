import torch
import torch.nn as nn

from smforge.cuda import build

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>
using namespace nvcuda;

// 64x64x32 tile per block of 4 warps; each warp owns a 32x32 tile of 2x2 16x16x16 fragments.
constexpr int BM = 64, BN = 64, BK = 32, PAD = 8;

// Two shared-memory stages: cp.async fills the next tile while the warps multiply this one.
__global__ void __launch_bounds__(128) gemm_wmma(const __half* __restrict__ A,
                                                 const __half* __restrict__ B,
                                                 __half* __restrict__ C, int M, int N, int K) {
  __shared__ __align__(16) __half As[2][BM][BK + PAD];
  __shared__ __align__(16) __half Bs[2][BK][BN + PAD];
  __shared__ __align__(16) float Cs[BM][BN + 4];
  int tid = threadIdx.x, warp = tid / 32;
  int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
  int wm = (warp / 2) * 32, wn = (warp % 2) * 32;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
  auto load = [&](int stage, int k0) {
    for (int v = tid; v < BM * BK / 8; v += 128) {
      int r = v / (BK / 8), c = (v % (BK / 8)) * 8;
      if (row0 + r < M && k0 + c < K)
        __pipeline_memcpy_async(&As[stage][r][c], A + (long)(row0 + r) * K + k0 + c, 16);
      else
        *(uint4*)&As[stage][r][c] = make_uint4(0, 0, 0, 0);
    }
    for (int v = tid; v < BK * BN / 8; v += 128) {
      int r = v / (BN / 8), c = (v % (BN / 8)) * 8;
      if (k0 + r < K && col0 + c < N)
        __pipeline_memcpy_async(&Bs[stage][r][c], B + (long)(k0 + r) * N + col0 + c, 16);
      else
        *(uint4*)&Bs[stage][r][c] = make_uint4(0, 0, 0, 0);
    }
    __pipeline_commit();
  };
  int tiles = (K + BK - 1) / BK;
  load(0, 0);
  for (int t = 0; t < tiles; ++t) {
    int s = t & 1;
    if (t + 1 < tiles) load(s ^ 1, (t + 1) * BK);
    else __pipeline_commit();
    __pipeline_wait_prior(1);
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b[2];
      for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], &As[s][wm + i * 16][kk], BK + PAD);
      for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], &Bs[s][kk][wn + j * 16], BN + PAD);
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      wmma::store_matrix_sync(&Cs[wm + i * 16][wn + j * 16], acc[i][j], BN + 4,
                              wmma::mem_row_major);
  __syncthreads();
  for (int e = tid; e < BM * BN; e += 128) {
    int r = e / BN, c = e % BN;
    if (row0 + r < M && col0 + c < N) C[(long)(row0 + r) * N + col0 + c] = __float2half(Cs[r][c]);
  }
}

int gemm(const void* a, const void* b, void* c, int m, int n, int k, cudaStream_t stream) {
  dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
  gemm_wmma<<<grid, 128, 0, stream>>>((const __half*)a, (const __half*)b, (__half*)c, m, n, k);
  return 0;
}
"""

CPP_SRC = r"""
int gemm(const void* a, const void* b, void* c, int m, int n, int k, cudaStream_t stream);

torch::Tensor forward(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.scalar_type() == torch::kHalf && b.scalar_type() == torch::kHalf, "fp16 inputs");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous());
  int m = a.size(0), k = a.size(1), n = b.size(1);
  TORCH_CHECK(k % 8 == 0 && n % 8 == 0, "K and N must be multiples of 8");
  auto c = torch::empty({m, n}, a.options());
  int status = gemm(a.data_ptr(), b.data_ptr(), c.data_ptr(), m, n, k,
                    at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == 0, "GEMM status ", status);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}
"""

ext = build(CUDA_SRC, CPP_SRC, ["forward"])


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return ext.forward(A.contiguous(), B.contiguous())
