"""A/A timing noise: the same call in both arms of an interleaved comparison (worker side).

Any speedup measured here is noise. The suggested minimum effect is twice the worst A/A
deviation, and never below 1%.
"""

from __future__ import annotations

import torch

from smforge.harness.timing import Timer
from smforge.models import NoiseCase, NoiseResult

MIN_EFFECT_FLOOR = 0.01


def measure_noise(timer: Timer, device: torch.device) -> NoiseResult:
    small = torch.randn(4096, device=device)
    src = torch.randn(64 << 20, device=device)
    dst = torch.empty_like(src)
    a = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    cases = {
        "launch_bound": lambda: small.add(1.0),
        "memory_bound": lambda: dst.copy_(src),
        "compute_bound": lambda: torch.mm(a, a),
    }
    results = []
    for name, fn in cases.items():
        comparison = timer.compare(fn, fn)
        results.append(
            NoiseCase(
                name=name,
                time_us=comparison.baseline_us,
                deviation=abs(comparison.speedup - 1),
                spread=max(abs(r - 1) for r in comparison.ratios),
            )
        )
    worst = max(case.deviation for case in results)
    return NoiseResult(cases=results, min_effect=max(MIN_EFFECT_FLOOR, round(2 * worst, 3)))
