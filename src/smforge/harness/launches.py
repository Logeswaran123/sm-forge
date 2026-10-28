"""The CUDA kernels one forward launches, from the PyTorch profiler (worker side).

A kernel counts as Triton when the module defines it or when Triton's runtime launched it during
the warm-up call, which also catches kernels defined inside functions.

On WSL2 the profiler stops recording CUDA activity about 40 s after a process first uses it, and
does not recover. An empty profile therefore raises ``ProfilerUnavailable``, which ends the
worker, instead of reading as a forward that launched nothing.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Callable
from types import ModuleType

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile
from triton.runtime.jit import JITFunction

from smforge.harness.audit import classify_launch
from smforge.harness.facts import capture_launches
from smforge.models import Launch, LaunchProfile


class ProfilerUnavailable(RuntimeError):
    fatal = True


def triton_names(module: ModuleType) -> set[str]:
    """Kernel names of the ``@triton.jit`` functions in ``module``, through autotune wrappers."""
    names = set()
    for value in vars(module).values():
        for _ in range(4):
            if isinstance(value, JITFunction) or not hasattr(value, "fn"):
                break
            value = value.fn
        if isinstance(value, JITFunction):
            names.add(value.fn.__name__)
    return names


def profile_launches(fn: Callable[[], object], names: set[str], reps: int = 3) -> LaunchProfile:
    with capture_launches() as records:
        fn()
        torch.cuda.synchronize()
    names = names | {kernel.name for kernel, _ in records}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(reps):
                fn()
            torch.cuda.synchronize()
        events = [e for e in prof.events() if e.device_type == DeviceType.CUDA]
    if not events:
        raise ProfilerUnavailable("the profiler recorded no CUDA activity")
    totals: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    for event in events:
        totals[event.name][0] += 1
        totals[event.name][1] += event.time_range.elapsed_us()
    launches = sorted(
        (
            Launch(
                name=name,
                category=classify_launch(name, names),
                calls=calls / reps,
                duration_us=total / reps,
            )
            for name, (calls, total) in totals.items()
        ),
        key=lambda launch: -launch.duration_us,
    )
    return LaunchProfile(launches=launches, kernel_us=sum(x.duration_us for x in launches))
