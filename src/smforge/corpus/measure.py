"""Reference measurements for the importers and the baselines (worker side)."""

from __future__ import annotations

from statistics import median
from types import ModuleType

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from smforge.harness.correctness import compare_outputs, flatten
from smforge.harness.tolerance import DEFAULT_TOLERANCES
from smforge.harness.workload import Workload
from smforge.models import ReferenceProfile

COMPILE_MODES = {
    "eager": None,
    "compile": "default",
    "max-autotune": "max-autotune-no-cudagraphs",
}


def profile_reference(workload: Workload, module: ModuleType) -> ReferenceProfile:
    """Footprint, time, FLOPs and bytes of one forward at the variant's dtype.

    The footprint is the peak during the forward: the resident model and inputs plus whatever the
    forward allocates, not the transient copies made while building the model.

    FLOPs are left out when PyTorch's counter rejects an op, as it does grouped-query attention.

    For a dtype below float32 the same weights and inputs also run in float32. ``stable`` says
    whether the low-precision result stays within that dtype's default tolerance of the float32
    one; when it does not, an accurate kernel would fail against the low-precision oracle.
    """
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    model = workload.build(module)
    inputs = workload.inputs()
    torch.cuda.reset_peak_memory_stats()
    flop = None
    try:
        with FlopCounterMode(display=False) as counter:
            output = model(*inputs)
        flop = counter.get_total_flops() or None
    except AssertionError:
        output = model(*inputs)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    outputs = [t for t in flatten(output) if isinstance(t, torch.Tensor)]
    finite = all(bool(t.isfinite().all()) for t in outputs if t.is_floating_point())
    tensors = [*_tensors(inputs), *outputs, *model.parameters(), *model.buffers()]
    profile = ReferenceProfile(
        peak_bytes=peak,
        time_us=_time(lambda: model(*inputs)),
        flop=flop,
        bytes_moved=sum(t.numel() * t.element_size() for t in tensors),
        finite=finite,
    )
    dtype = workload.spec.dtype(workload.variant) if workload.spec else "float32"
    if dtype in DEFAULT_TOLERANCES and dtype not in ("float32", "float64"):
        reference = workload.build(module, dtype=torch.float32)
        expected = reference(*[x.float() if _floating(x) else x for x in inputs])
        diffs, error = compare_outputs(expected, output, *DEFAULT_TOLERANCES[dtype])
        profile.stable = error is None and not any(d.mismatched for d in diffs)
        profile.max_abs_error = max((d.max_abs for d in diffs), default=None)
    return profile


def compiled(model: nn.Module, mode: str) -> nn.Module:
    if mode not in COMPILE_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(COMPILE_MODES)}")
    if COMPILE_MODES[mode] is None:
        return model
    torch._dynamo.reset()
    return torch.compile(model, mode=COMPILE_MODES[mode])


def _time(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for start, end in events:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return median(start.elapsed_time(end) * 1e3 for start, end in events)


def _tensors(values: list) -> list[torch.Tensor]:
    return [v for v in values if isinstance(v, torch.Tensor)]


def _floating(value) -> bool:
    return isinstance(value, torch.Tensor) and value.is_floating_point()
