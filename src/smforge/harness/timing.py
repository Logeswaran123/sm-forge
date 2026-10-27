"""Device timing with CUDA events (worker side).

Every sample is preceded, untimed, by an L2 flush and a 1024x1024 matmul. The matmul keeps the
GPU busy while the host enqueues the start event, the call and the end event, so a short kernel
is not timed together with its launch latency.

A run is summarized by a low quantile of its samples (``TimingConfig.quantile``, p20 by
default), not the median. Interference from other work on the GPU only ever adds time; on WSL2,
where the Windows desktop shares the GPU, it slowed up to 40% of samples of a 1.2 ms matmul,
and the median of identical runs moved by 16%. The p20 of the same runs moved by under 0.1%.

``Timer.compare`` times two calls in alternating rounds. Drift in clocks or temperature then
lands on both arms, and identical code measures as parity instead of a speedup.
"""

from __future__ import annotations

from collections.abc import Callable
from statistics import fmean, median

import torch

from smforge.config import TimingConfig
from smforge.models import Comparison, Timing, Verdict


class Timer:
    def __init__(self, config: TimingConfig, device: torch.device):
        self.config = config
        self._flush = torch.empty(config.flush_mb << 20, dtype=torch.int8, device=device)
        self._fill = torch.randn(1024, 1024, device=device)

    def samples(self, fn: Callable[[], object], n: int, flush: bool = True) -> list[float]:
        """``n`` device timings of ``fn`` in microseconds."""
        events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(n)
        ]
        for start, end in events:
            if flush:
                self._flush.zero_()
            torch.mm(self._fill, self._fill)
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize()
        return [start.elapsed_time(end) * 1e3 for start, end in events]

    def warmup(self, fn: Callable[[], object], n: int | None = None) -> None:
        for _ in range(self.config.warmup if n is None else n):
            fn()
        torch.cuda.synchronize()

    def time(self, fn: Callable[[], object], flush: bool = True) -> Timing:
        self.warmup(fn)
        return summarize(self.samples(fn, self.config.iters, flush), self.config.quantile)

    def compare(
        self, baseline: Callable[[], object], candidate: Callable[[], object]
    ) -> Comparison:
        self.warmup(baseline)
        self.warmup(candidate)
        q = self.config.quantile
        per_round = max(1, self.config.iters // self.config.rounds)
        baseline_all: list[float] = []
        candidate_all: list[float] = []
        ratios = []
        for r in range(self.config.rounds):
            if r % 2 == 0:
                b = self.samples(baseline, per_round)
                c = self.samples(candidate, per_round)
            else:
                c = self.samples(candidate, per_round)
                b = self.samples(baseline, per_round)
            baseline_all += b
            candidate_all += c
            ratios.append(quantile(b, q) / quantile(c, q))
        baseline_us, candidate_us = quantile(baseline_all, q), quantile(candidate_all, q)
        speedup = baseline_us / candidate_us
        return Comparison(
            baseline_us=baseline_us,
            candidate_us=candidate_us,
            speedup=speedup,
            verdict=verdict(speedup, self.config.min_effect),
            min_effect=self.config.min_effect,
            quantile=q,
            ratios=ratios,
        )


def quantile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    return ordered[round(q * (len(ordered) - 1))]


def summarize(samples: list[float], q: float) -> Timing:
    return Timing(
        time_us=quantile(samples, q),
        quantile=q,
        median_us=median(samples),
        mean_us=fmean(samples),
        min_us=min(samples),
        max_us=max(samples),
        samples=len(samples),
    )


def verdict(speedup: float, min_effect: float) -> Verdict:
    if speedup >= 1 + min_effect:
        return Verdict.FASTER
    if speedup <= 1 / (1 + min_effect):
        return Verdict.SLOWER
    return Verdict.PARITY


def graph_callable(fn: Callable[[], object]) -> Callable[[], None]:
    """Capture ``fn`` in a CUDA graph and return its replay."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph.replay
