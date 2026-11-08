# Kernel format

sm-forge uses the same file layout as KernelBench and Xe-Forge. A problem is three files:

| File | What it holds |
|---|---|
| `reference.py` | `class Model(nn.Module)` computing the result with PyTorch. This is the oracle. |
| `kernel.py` | The same `Model`, with `forward` launching your own kernels (Triton or CUDA C++). |
| `spec.yaml` | Input shapes and dtypes, `Model.__init__` arguments, and the variants to check and time. |

The kernel's `Model.__init__` takes the reference's arguments and creates the same parameters
and buffers, so the reference's weights load into it.

## The spec

```yaml
inputs:
  X: {shape: [M, K], dtype: float16}
inits:
  - in_features: K
  - out_features: N
ci:                                   # small, odd sizes: correctness only
  - params: [X]
    dims: {M: 127, K: 96, N: 200}
bench-gpu:                            # the shape that is timed
  - params: [X]
    dtype: float16
    dims: {M: 4096, K: 2048, N: 2048}
    flop: "2*M*N*K"
    bytes: "2*(M*K + N*K + M*N)"
```

`inputs` gives each input's shape in size letters and its dtype. `inits` lists the arguments of
`Model.__init__` in order. Each variant (`ci`, `bench-gpu`, and any others you add) is a list of
cases: `params` names the inputs, `dims` sets the size letters, and a `dtype` overrides the
inputs'. `ci` is only checked for correctness, so use small, odd sizes there to catch edge
cases. `bench-gpu` is checked and timed. `flop` and `bytes` give TFLOPS and GB/s in the timing
output. Tolerances come from the command line, then the variant (`rtol`, `atol`), then the
environment (`CORRECTNESS_RTOL`, `CORRECTNESS_ATOL`), then a default for the variant's dtype.

`tests/kernels/linear_relu/` is a complete example with both kernel languages.

## Triton kernels

`forward` launches `@triton.jit` kernels. It and the helpers it calls may only allocate tensors
and change views, layouts and dtypes (`torch.empty`, `.view`, `.reshape`, `.permute`,
`.contiguous`, `.to`). Any PyTorch compute (`torch.matmul`, `F.*`, `.sum()`, `@`, calling an
`nn` layer) is rejected.

## CUDA C++ kernels

The file is the same Python module. The device code and the PyTorch binding are two strings,
built when the module is imported:

```python
import torch
import torch.nn as nn

from smforge.cuda import build

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void __launch_bounds__(256) scale_kernel(const T* x, T* y, float s, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i < n) y[i] = T(float(x[i]) * s);
}

void scale(const void* x, void* y, float s, long n, cudaStream_t stream) {
  scale_kernel<__half><<<(n + 255) / 256, 256, 0, stream>>>((const __half*)x, (__half*)y, s, n);
}
"""

CPP_SRC = r"""
void scale(const void* x, void* y, float s, long n, cudaStream_t stream);

torch::Tensor forward(torch::Tensor x, double s) {
  auto y = torch::empty_like(x);
  scale(x.data_ptr(), y.data_ptr(), s, x.numel(), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
"""

ext = build(CUDA_SRC, CPP_SRC, ["forward"])


class Model(nn.Module):
    def forward(self, x):
        return ext.forward(x, 2.0)
```

- `CUDA_SRC` is compiled by nvcc for this GPU (`-arch=sm_120a` on an RTX 5080) and must not
  include PyTorch headers. Its launchers take raw pointers, sizes and a `cudaStream_t`.
- `CPP_SRC` is compiled with `torch/extension.h` already included. It unpacks the tensors,
  allocates outputs, calls the launchers on `at::cuda::getCurrentCUDAStream()` and checks the
  launch. It may not compute with PyTorch.
- `build` exposes the listed functions as `ext.<name>`.

Rules the static checks enforce on CUDA files: a launch-error check after `<<<...>>>` launches
(`C10_CUDA_KERNEL_LAUNCH_CHECK()` or `cudaGetLastError()`), no `cudaMalloc` or `cudaFree` per call,
no `static` variables, no PyTorch headers in `CUDA_SRC`. They warn about kernels without
`__launch_bounds__` or a bounds check, launches on the default stream, and host syncs.

### Builds

The device code compiles with nvcc alone, and the binding against a precompiled
`torch/extension.h`, so a new kernel builds in about 5 s instead of 30. Builds are cached in
`.smforge/cuda/` by a hash of the sources, flags, GPU and PyTorch version, and a cached kernel
loads at once. The precompiled header takes about 730 MB there.

A compile error is reported with its line in the kernel file:

```
CUDA build failed:
line 13: identifier "undefined_name" is undefined
```

A source that includes `cutlass/` gets the CUTLASS headers: a sparse git checkout of v4.8.1 in
`.smforge/cutlass/` (37 MB), made on first use, or the checkout `CUTLASS_DIR` points to. On sm_120
use CUTLASS's SM80 kernels for fp16 and bf16, since its SM120 builders take only fp8 and below.

`src/smforge/llm/prompts/cuda/format.md` is the same layout as the LLM sees it.
