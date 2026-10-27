"""Facts about the Triton kernels a forward launches: resources, SASS mix, occupancy (worker side).

Triton's launch hooks do not see the grid, so ``capture_launches`` wraps ``CompiledKernel.run``
for the duration of one forward and records every launch with its compiled kernel.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path

import torch
import triton
from triton.compiler.compiler import CompiledKernel

from smforge.models import KernelFacts

INSTRUCTIONS = (
    "FFMA",
    "HFMA2",
    "LDGSTS",
    "UTMALDG",
    "UTMASTG",
    "LDG",
    "STG",
    "LDS",
    "STS",
    "LDSM",
    "LDL",
    "STL",
    "BAR",
    "SHFL",
)
_SASS_LINE = re.compile(r"^\s*/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P[T\d]+\s+)?([A-Z][A-Z0-9_]*)")
_histograms: dict[str, Counter[str]] = {}

Record = tuple[CompiledKernel, tuple[int, int, int]]


@contextmanager
def capture_launches() -> Iterator[list[Record]]:
    records: list[Record] = []
    original = CompiledKernel.run

    def run(self: CompiledKernel):
        launcher = original.fget(self)

        def launch(grid_x, grid_y, grid_z, *args):
            records.append((self, (grid_x, grid_y, grid_z)))
            return launcher(grid_x, grid_y, grid_z, *args)

        return launch

    CompiledKernel.run = property(run)
    try:
        yield records
    finally:
        CompiledKernel.run = original


@cache
def nvdisasm() -> str:
    bundled = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "nvdisasm"
    if bundled.exists():
        return str(bundled)
    found = shutil.which("nvdisasm")
    if found is None:
        raise FileNotFoundError("nvdisasm not found; install the CUDA toolkit")
    return found


def sass_histogram(kernel: CompiledKernel) -> Counter[str]:
    """Opcode counts of the kernel's SASS."""
    if kernel.hash not in _histograms:
        with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
            f.write(kernel.asm["cubin"])
            f.flush()
            out = subprocess.run(
                [nvdisasm(), "-c", f.name], capture_output=True, text=True, check=True
            ).stdout
        _histograms[kernel.hash] = Counter(
            m.group(1) for line in out.splitlines() if (m := _SASS_LINE.match(line))
        )
    return _histograms[kernel.hash]


def ctas_per_sm(kernel: CompiledKernel) -> int | None:
    """Resident CTAs per SM, from the driver's occupancy calculator."""
    try:
        from cuda.bindings import driver
    except ImportError:
        return None
    err, blocks = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        driver.CUfunction(kernel.function),
        kernel.metadata.num_warps * 32,
        kernel.metadata.shared,
    )
    return blocks if err == driver.CUresult.CUDA_SUCCESS else None


def collect_facts(fn: Callable[[], object], sm_count: int) -> list[KernelFacts]:
    with capture_launches() as records:
        fn()
        torch.cuda.synchronize()
    facts = []
    seen = set()
    for kernel, grid in records:
        if (kernel.hash, grid) in seen:
            continue
        seen.add((kernel.hash, grid))
        histogram = sass_histogram(kernel)
        ctas = grid[0] * grid[1] * grid[2]
        resident = ctas_per_sm(kernel)
        facts.append(
            KernelFacts(
                name=kernel.name,
                grid=list(grid),
                ctas=ctas,
                num_warps=kernel.metadata.num_warps,
                num_stages=kernel.metadata.num_stages,
                n_regs=kernel.n_regs,
                n_spills=kernel.n_spills,
                shared_bytes=kernel.metadata.shared,
                ctas_per_sm=resident,
                waves=ctas / (sm_count * resident) if resident else None,
                mma={op: n for op, n in histogram.items() if "MMA" in op},
                instructions={op: histogram[op] for op in INSTRUCTIONS if histogram[op]},
                total_instructions=sum(histogram.values()),
            )
        )
    return facts
