import torch
import torch.nn as nn

from smforge.cuda import build

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

#define ROWS 8          // query rows per block (one warp each)
#define BK 32           // keys per tile
#define MAXD 128
#define KSTRIDE (MAXD + 1)

template <typename T>
__global__ void __launch_bounds__(256) attn_kernel(
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
    T* __restrict__ o, int B, int Hq, int Hkv, int Lq, int Lk, int D, float scale) {
  __shared__ float Ks[BK * KSTRIDE];
  __shared__ float Vs[BK * MAXD];
  __shared__ float Qs[ROWS * MAXD];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int bh = blockIdx.y;
  const int b = bh / Hq;
  const int h = bh % Hq;
  const int group = Hq / Hkv;
  const int hk = h / group;
  const int r0 = blockIdx.x * ROWS;
  const int row = r0 + warp;

  const T* qp = q + ((long)b * Hq + h) * (long)Lq * D;
  const T* kp = k + ((long)b * Hkv + hk) * (long)Lk * D;
  const T* vp = v + ((long)b * Hkv + hk) * (long)Lk * D;
  T* op = o + ((long)b * Hq + h) * (long)Lq * D;

  // load Q rows
  for (int idx = tid; idx < ROWS * MAXD; idx += blockDim.x) {
    int r = idx / MAXD, d = idx % MAXD;
    int gr = r0 + r;
    float val = 0.f;
    if (gr < Lq && d < D) val = float(qp[(long)gr * D + d]);
    Qs[idx] = val;
  }

  float acc[MAXD / 32];
#pragma unroll
  for (int t = 0; t < MAXD / 32; ++t) acc[t] = 0.f;
  float m = -INFINITY, l = 0.f;

  int last_row = r0 + ROWS - 1;
  if (last_row > Lq - 1) last_row = Lq - 1;
  int kend = last_row + 1;
  if (kend > Lk) kend = Lk;

  for (int kt = 0; kt < kend; kt += BK) {
    __syncthreads();
    for (int idx = tid; idx < BK * MAXD; idx += blockDim.x) {
      int j = idx / MAXD, d = idx % MAXD;
      int gj = kt + j;
      float kv = 0.f, vv = 0.f;
      if (gj < Lk && d < D) {
        kv = float(kp[(long)gj * D + d]);
        vv = float(vp[(long)gj * D + d]);
      }
      Ks[j * KSTRIDE + d] = kv;
      Vs[j * MAXD + d] = vv;
    }
    __syncthreads();

    // score for key kt+lane
    int gj = kt + lane;
    float s = -INFINITY;
    if (gj < Lk && gj <= row && row < Lq) {
      float dot = 0.f;
      const float* qr = Qs + warp * MAXD;
      const float* kr = Ks + lane * KSTRIDE;
      for (int d = 0; d < D; ++d) dot += qr[d] * kr[d];
      s = dot * scale;
    }
    float tmax = s;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffff, tmax, off));
    float m_new = fmaxf(m, tmax);
    if (m_new == -INFINITY) continue;  // nothing valid yet for this row (uniform per warp)
    float p = (s == -INFINITY) ? 0.f : __expf(s - m_new);
    float corr = (m == -INFINITY) ? 0.f : __expf(m - m_new);
    float psum = p;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) psum += __shfl_xor_sync(0xffffffff, psum, off);
    l = l * corr + psum;
    m = m_new;
#pragma unroll
    for (int t = 0; t < MAXD / 32; ++t) acc[t] *= corr;
    for (int j = 0; j < BK; ++j) {
      float pj = __shfl_sync(0xffffffff, p, j);
#pragma unroll
      for (int t = 0; t < MAXD / 32; ++t) {
        int d = lane + 32 * t;
        acc[t] += pj * Vs[j * MAXD + d];
      }
    }
  }

  if (row < Lq) {
    float inv = (l > 0.f) ? 1.f / l : 0.f;
#pragma unroll
    for (int t = 0; t < MAXD / 32; ++t) {
      int d = lane + 32 * t;
      if (d < D) op[(long)row * D + d] = T(acc[t] * inv);
    }
  }
}

template <typename T>
static void launch(const void* q, const void* k, const void* v, void* o, int B, int Hq, int Hkv,
                   int Lq, int Lk, int D, float scale, cudaStream_t stream) {
  dim3 grid((Lq + ROWS - 1) / ROWS, B * Hq);
  attn_kernel<T><<<grid, 256, 0, stream>>>((const T*)q, (const T*)k, (const T*)v, (T*)o,
                                           B, Hq, Hkv, Lq, Lk, D, scale);
}

void attn(int dtype, const void* q, const void* k, const void* v, void* o, int B, int Hq, int Hkv,
          int Lq, int Lk, int D, float scale, cudaStream_t stream) {
  if (dtype == 1) launch<__half>(q, k, v, o, B, Hq, Hkv, Lq, Lk, D, scale, stream);
  else if (dtype == 2) launch<__nv_bfloat16>(q, k, v, o, B, Hq, Hkv, Lq, Lk, D, scale, stream);
  else launch<float>(q, k, v, o, B, Hq, Hkv, Lq, Lk, D, scale, stream);
}
"""

CPP_SRC = r"""
void attn(int dtype, const void* q, const void* k, const void* v, void* o, int B, int Hq, int Hkv,
          int Lq, int Lk, int D, float scale, cudaStream_t stream);

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4);
  q = q.contiguous(); k = k.contiguous(); v = v.contiguous();
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type());
  int B = q.size(0), Hq = q.size(1), Lq = q.size(2), D = q.size(3);
  int Hkv = k.size(1), Lk = k.size(2);
  TORCH_CHECK(k.size(0) == B && v.size(0) == B && v.size(1) == Hkv && v.size(2) == Lk);
  TORCH_CHECK(k.size(3) == D && v.size(3) == D, "head dims must match");
  TORCH_CHECK(D <= 128, "head dim must be <= 128");
  TORCH_CHECK(Hkv > 0 && Hq % Hkv == 0);
  auto type = q.scalar_type();
  int dtype = type == torch::kHalf ? 1 : type == torch::kBFloat16 ? 2 : 0;
  auto o = torch::empty_like(q);
  if (o.numel() == 0) return o;
  float scale = 1.0f / sqrtf((float)D);
  attn(dtype, q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(), B, Hq, Hkv, Lq, Lk, D,
       scale, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return o;
}
"""

ext = build(CUDA_SRC, CPP_SRC, ["forward"])


class Model(nn.Module):
    def forward(self, q, k, v):
        return ext.forward(q.contiguous(), k.contiguous(), v.contiguous())
