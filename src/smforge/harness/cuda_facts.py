"""Facts about the CUDA kernels a forward launches from ``smforge.cuda`` extensions (worker side).

A kernel is the file's own when its symbol is in an extension built or loaded in this process.
The profiler names a kernel as GNU c++filt demangles it, or by its mangled name where c++filt
gives up (long CUTLASS templates), so both forms identify it; cu++filt names those for display. Grid, block, registers and shared memory (static and dynamic) come from
the profiler's trace of one forward; spills (local memory per thread) and the SASS mix from
cuobjdump on the extension. Resident CTAs per SM are computed from the device limits.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from smforge.cuda import function_name, loaded
from smforge.device.tools import find_tool
from smforge.harness.facts import _SASS_LINE, INSTRUCTIONS
from smforge.models import KernelFacts

RESERVED_SMEM = 1024
REGISTER_UNIT = 256
_FUNCTION = re.compile(r"^\s*Function\s*:?\s*(\S+?):?\s*$")
_USAGE = re.compile(r"(REG|STACK|SHARED|LOCAL):(\d+)")


@dataclass(frozen=True)
class Symbol:
    library: Path
    mangled: str
    demangled: str
    display: str

    @property
    def short(self) -> str:
        return function_name(self.display)


@cache
def symbols(library: Path) -> tuple[Symbol, ...]:
    """The kernels (CUDA entry functions) in ``library``."""
    out = _run("cuobjdump", "-symbols", str(library))
    mangled = [line.split()[-1] for line in out.splitlines() if "STO_ENTRY" in line]
    demangled = _demangle("c++filt", mangled)
    display = [
        nvidia if gnu == name else gnu
        for name, gnu, nvidia in zip(
            mangled, demangled, _demangle("cu++filt", mangled), strict=True
        )
    ]
    return tuple(Symbol(library, *names) for names in zip(mangled, demangled, display, strict=True))


def _demangle(tool: str, names: list[str]) -> list[str]:
    binary = find_tool(tool)
    if binary is None or not names:
        return list(names)
    out = subprocess.run([binary], input="\n".join(names), capture_output=True, text=True)
    demangled = out.stdout.splitlines()
    return demangled if len(demangled) == len(names) else list(names)


def own_kernels() -> dict[str, Symbol]:
    """Profiler name (demangled or mangled) to symbol, for every loaded extension."""
    found = {}
    for library in loaded().values():
        for symbol in symbols(library):
            found[symbol.mangled] = symbol
            found[symbol.demangled] = symbol
    return found


@cache
def resource_usage(library: Path) -> dict[str, dict[str, int]]:
    """Per mangled kernel name: REG, STACK, SHARED (static) and LOCAL bytes."""
    usage, current = {}, None
    for line in _run("cuobjdump", "-res-usage", str(library)).splitlines():
        if match := _FUNCTION.match(line):
            current = match.group(1)
        elif current and (values := _USAGE.findall(line)):
            usage[current] = {key: int(value) for key, value in values}
            current = None
    return usage


@cache
def sass_histograms(library: Path) -> dict[str, Counter[str]]:
    histograms: dict[str, Counter[str]] = {}
    current = None
    for line in _run("cuobjdump", "-sass", str(library)).splitlines():
        if match := _FUNCTION.match(line):
            current = histograms.setdefault(match.group(1), Counter())
        elif current is not None and (match := _SASS_LINE.match(line)):
            current[match.group(1)] += 1
    return histograms


def trace_launches(fn: Callable[[], object]) -> list[dict]:
    """Kernel events of one call of ``fn`` from the profiler's trace: name and launch args."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            prof.export_chrome_trace(f.name)
            trace = json.loads(Path(f.name).read_text())
    return [e for e in trace.get("traceEvents", []) if e.get("cat") == "kernel"]


def resident_ctas(regs: int, threads: int, smem: int, device: int = 0) -> int:
    """CTAs of this shape one SM holds at once."""
    props = torch.cuda.get_device_properties(device)
    warps = math.ceil(threads / props.warp_size)
    per_warp = math.ceil(max(regs, 1) * props.warp_size / REGISTER_UNIT) * REGISTER_UNIT
    limits = [
        props.max_threads_per_multi_processor // max(threads, 1),
        props.regs_per_multiprocessor // per_warp // max(warps, 1),
        props.shared_memory_per_multiprocessor // (smem + RESERVED_SMEM),
    ]
    if blocks := _max_blocks_per_sm(device):
        limits.append(blocks)
    return max(min(limits), 0)


def collect_cuda_facts(fn: Callable[[], object], sm_count: int) -> list[KernelFacts]:
    own = own_kernels()
    if not own:
        return []
    events = trace_launches(fn)
    if not events:
        from smforge.harness.launches import ProfilerUnavailable

        raise ProfilerUnavailable("the profiler recorded no CUDA activity")
    facts, seen = [], set()
    for event in events:
        symbol = own.get(event["name"])
        if symbol is None:
            continue
        args = event.get("args", {})
        grid, block = args.get("grid", [1, 1, 1]), args.get("block", [1, 1, 1])
        if (symbol.mangled, tuple(grid)) in seen:
            continue
        seen.add((symbol.mangled, tuple(grid)))
        threads = block[0] * block[1] * block[2]
        regs = int(args.get("registers per thread", 0))
        smem = int(args.get("shared memory", 0))
        usage = resource_usage(symbol.library).get(symbol.mangled, {})
        histogram = sass_histograms(symbol.library).get(symbol.mangled, Counter())
        ctas = grid[0] * grid[1] * grid[2]
        resident = resident_ctas(regs, threads, smem)
        facts.append(
            KernelFacts(
                name=symbol.short,
                grid=list(grid),
                ctas=ctas,
                num_warps=math.ceil(threads / 32),
                num_stages=None,
                n_regs=regs or usage.get("REG", 0),
                n_spills=usage.get("LOCAL", 0) // 4,
                shared_bytes=smem,
                ctas_per_sm=resident or None,
                waves=ctas / (sm_count * resident) if resident else None,
                mma={op: n for op, n in histogram.items() if "MMA" in op},
                instructions={op: histogram[op] for op in INSTRUCTIONS if histogram[op]},
                total_instructions=sum(histogram.values()),
                block=list(block),
                symbol=symbol.mangled,
            )
        )
    return facts


@cache
def _max_blocks_per_sm(device: int) -> int | None:
    try:
        from cuda.bindings import driver
    except ImportError:
        return None
    attribute = driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR
    err, value = driver.cuDeviceGetAttribute(attribute, device)
    return value if err == driver.CUresult.CUDA_SUCCESS else None


def _run(tool: str, *args: str) -> str:
    binary = find_tool(tool)
    if binary is None:
        raise FileNotFoundError(f"{tool} not found; install the CUDA toolkit")
    return subprocess.run([binary, *args], capture_output=True, text=True, check=True).stdout
