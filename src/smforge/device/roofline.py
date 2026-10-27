"""Measured bandwidth and matmul throughput, the ceilings of the roofline (worker side).

Matmul peaks come from cuBLAS through PyTorch, the best a library reaches on this part rather
than the datasheet figure.
"""

from __future__ import annotations

import torch
import triton

from smforge.config import TimingConfig
from smforge.device.kernels import read_kernel
from smforge.harness.timing import Timer
from smforge.models import Roofline

MATMUL_SIZES = (4096, 8192)
DRAM_BYTES = 1 << 30
L2_BYTES = 16 << 20
READ_BLOCK = 4096
TIMING = TimingConfig(warmup=10, iters=30, rounds=1, flush_mb=256)
MATMUL_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "tf32": torch.float32,
}


def measure_roofline(device: torch.device) -> Roofline:
    timer = Timer(TIMING, device)
    return Roofline(
        dram_copy_gbps=_copy_gbps(timer, device, DRAM_BYTES),
        dram_read_gbps=_read_gbps(timer, device, DRAM_BYTES),
        l2_copy_gbps=_copy_gbps(timer, device, L2_BYTES, flush=False),
        tflops=_matmul_tflops(timer, device),
    )


def _copy_gbps(timer: Timer, device: torch.device, nbytes: int, flush: bool = True) -> float:
    src = torch.rand(nbytes // 4, device=device)
    dst = torch.empty_like(src)
    timing = timer.time(lambda: dst.copy_(src), flush=flush)
    return 2 * nbytes / timing.median_us / 1e3


def _read_gbps(timer: Timer, device: torch.device, nbytes: int) -> float:
    n = nbytes // 4
    x = torch.rand(n, device=device)
    grid = (triton.cdiv(n, READ_BLOCK),)
    out = torch.empty(grid[0], device=device)
    timing = timer.time(lambda: read_kernel[grid](x, out, n, BLOCK=READ_BLOCK))
    return nbytes / timing.median_us / 1e3


def _matmul_tflops(timer: Timer, device: torch.device) -> dict[str, float]:
    tflops = {}
    for name, dtype in MATMUL_DTYPES.items():
        torch.backends.cuda.matmul.allow_tf32 = name == "tf32"
        try:
            tflops[name] = max(_mm_tflops(timer, device, dtype, n) for n in MATMUL_SIZES)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = False
    try:
        tflops["float8_e4m3fn"] = max(_fp8_tflops(timer, device, n) for n in MATMUL_SIZES)
    except RuntimeError:
        pass
    return tflops


def _mm_tflops(timer: Timer, device: torch.device, dtype: torch.dtype, n: int) -> float:
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    timing = timer.time(lambda: torch.mm(a, b), flush=False)
    return 2 * n**3 / timing.median_us / 1e6


def _fp8_tflops(timer: Timer, device: torch.device, n: int) -> float:
    a = torch.randn(n, n, device=device).to(torch.float8_e4m3fn)
    b = torch.randn(n, n, device=device).to(torch.float8_e4m3fn).t()
    one = torch.ones((), device=device)
    timing = timer.time(
        lambda: torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16),
        flush=False,
    )
    return 2 * n**3 / timing.median_us / 1e6
