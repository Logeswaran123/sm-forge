The file is Python. It keeps the CUDA code in one string, the PyTorch binding in another, and
builds them when it is imported:

```python
import torch
import torch.nn as nn

from smforge.cuda import build

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void __launch_bounds__(256) scale_kernel(const T* __restrict__ x, T* __restrict__ y,
                                                    float s, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i < n) y[i] = T(float(x[i]) * s);
}

template <typename T>
static void launch(const void* x, void* y, float s, long n, cudaStream_t stream) {
  scale_kernel<T><<<(n + 255) / 256, 256, 0, stream>>>((const T*)x, (T*)y, s, n);
}

void scale(int dtype, const void* x, void* y, float s, long n, cudaStream_t stream) {
  if (dtype == 1) launch<__half>(x, y, s, n, stream);
  else if (dtype == 2) launch<__nv_bfloat16>(x, y, s, n, stream);
  else launch<float>(x, y, s, n, stream);
}
"""

CPP_SRC = r"""
void scale(int dtype, const void* x, void* y, float s, long n, cudaStream_t stream);

torch::Tensor forward(torch::Tensor x, double s) {
  TORCH_CHECK(x.is_contiguous());
  auto type = x.scalar_type();
  int dtype = type == torch::kHalf ? 1 : type == torch::kBFloat16 ? 2 : 0;
  auto y = torch::empty_like(x);
  scale(dtype, x.data_ptr(), y.data_ptr(), s, x.numel(), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
"""

ext = build(CUDA_SRC, CPP_SRC, ["forward"])
```

- `CUDA_SRC` is compiled by nvcc for `$arch` and must not include PyTorch headers (`torch/`,
  `ATen/`, `c10/`). Its host functions take raw pointers, sizes and a `cudaStream_t`, and launch
  on that stream. CUDA's half and bfloat16 operators and conversions are enabled. CUTLASS
  $cutlass headers are available (`#include "cutlass/gemm/device/gemm.h"`); on this GPU use its
  SM80 kernels (`cutlass::arch::Sm80`) for fp16 and bf16, within $smem KB of shared memory per
  block.
- `CPP_SRC` is compiled with `torch/extension.h` and `ATen/cuda/CUDAContext.h` already included.
  It declares the `CUDA_SRC` host functions, checks and unpacks tensors (`data_ptr`, `size`,
  `stride`, `scalar_type`), allocates outputs with `torch::empty`, calls the host functions with
  `at::cuda::getCurrentCUDAStream()`, and checks the launches with
  `C10_CUDA_KERNEL_LAUNCH_CHECK()`. It may not compute with PyTorch (`torch::matmul`,
  `x.sum()`, ...).
- `build(CUDA_SRC, CPP_SRC, [...])` exposes the listed `CPP_SRC` functions as `ext.<name>`.
  Builds take a few seconds and are cached; a compile error comes back with its line in the file.
- Give every `__global__` kernel `__launch_bounds__` equal to its threads per block. No
  `cudaMalloc` or `cudaFree` (allocate scratch with `torch::empty` in the binding), no `static`
  variables, no host synchronization.
