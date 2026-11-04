"""Evaluation: the pipeline over a split with one fixed config, and what to report about it.

``evaluate`` runs ``run_pipeline`` on a problem (``best_of`` times, keeping the fastest final)
and then measures the final kernel past what the pipeline checked:

- correctness on fresh input draws (from ``FRESH_DRAW``) on ``ci`` and ``bench-gpu``, and on the
  problem's original KernelBench shapes (the ``kernelbench`` variant at the timed dtype), which no
  check during optimization saw;
- the forward speedup over the seed (the slowest of the final A/B runs) and the kernel-only one
  (the sum of the forward's kernel times from the profiler);
- speedups over eager PyTorch, ``torch.compile`` and ``torch.compile`` max-autotune, from the
  baseline table;
- the fraction of the roofline: the time the measured peaks allow for the spec's FLOPs and bytes,
  over the final time. The peaks are measured, not the hardware's limits, and a good GEMM can run
  a few percent above them. Above ``ABOVE_ROOF`` the kernel does less work than the reference's
  FLOP count.
- LLM calls, tokens, the cache-hit share of input tokens, dollars and wall time.

Each problem's result is one JSON file in the run's directory, with the run's config beside it;
``summarize`` aggregates a run, and ``report`` writes the Markdown tables for several runs.
"""

from __future__ import annotations

import math
import statistics
import time
from pathlib import Path

from pydantic import BaseModel, Field

from smforge.agents.analyze import TIMED_VARIANT
from smforge.agents.pipeline import Options, PipelineResult, run_pipeline
from smforge.corpus.catalog import Problem
from smforge.corpus.results import RESULTS, load_results
from smforge.harness.runner import Runner, WorkerError
from smforge.kb.loader import KnowledgeBase
from smforge.llm.client import Completer, Usage
from smforge.models import DeviceProfile
from smforge.spec import load_spec

FRESH_DRAW = 1000
HELD_OUT = "kernelbench"
ABOVE_ROOF = 1.15
BASELINES = ("eager", "compile", "max-autotune")


class RunConfig(BaseModel):
    """Everything a run's results depend on; a run directory holds one."""

    split: str | None
    dsl: str
    model: str
    effort: str
    max_iters: int
    max_stages: int | None
    best_of: int = 1
    kb: bool = True
    arch_gating: bool = True
    facts: bool = True
    reanalysis: bool = True
    rollback: bool = True
    ncu: bool = True
    memcheck: bool = True

    @property
    def options(self) -> Options:
        return Options(facts=self.facts, reanalysis=self.reanalysis, rollback=self.rollback)


class LLMUse(BaseModel):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def cache_hit_share(self) -> float | None:
        total = self.input_tokens + self.cache_read_tokens + self.cache_write_tokens
        return self.cache_read_tokens / total if total else None


class ProblemEval(BaseModel):
    problem: str
    family: str
    seed_us: float
    final_us: float
    ab: list[float] = Field(default_factory=list)
    kept_seed: bool
    plan: list[str] = Field(default_factory=list)
    accepted: list[str] = Field(default_factory=list)
    runs: list[float] = Field(default_factory=list, description="each best-of run's speedup")
    baselines_us: dict[str, float] = Field(default_factory=dict)
    seed_kernel_us: float | None = None
    final_kernel_us: float | None = None
    roofline_us: float | None = None
    correctness: dict[str, str] = Field(default_factory=dict)
    llm: LLMUse = Field(default_factory=LLMUse)
    wall_s: float = 0.0

    @property
    def speedup(self) -> float:
        return self.seed_us / self.final_us

    @property
    def kernel_speedup(self) -> float | None:
        if self.seed_kernel_us and self.final_kernel_us:
            return self.seed_kernel_us / self.final_kernel_us
        return None

    @property
    def roofline_fraction(self) -> float | None:
        return self.roofline_us / self.final_us if self.roofline_us else None

    def over(self, baseline: str) -> float | None:
        time_us = self.baselines_us.get(baseline)
        return time_us / self.final_us if time_us else None

    @property
    def correct(self) -> bool:
        return not any(v.startswith("fail") for v in self.correctness.values())


def evaluate(
    problem: Problem,
    runner: Runner,
    llm: Completer,
    kb: KnowledgeBase,
    profile: DeviceProfile | None,
    config: RunConfig,
    workdir: Path,
    keep: int | None = 2,
) -> ProblemEval:
    start = time.monotonic()
    usage = Usage()
    results: list[PipelineResult] = []
    for attempt in range(config.best_of):
        result = run_pipeline(
            problem,
            runner,
            llm,
            kb,
            profile,
            config.max_iters,
            workdir / (f"run-{attempt}" if config.best_of > 1 else "pipeline"),
            keep,
            config.max_stages,
            config.ncu,
            config.memcheck,
            None,
            config.options,
        )
        usage.add(result.usage)
        results.append(result)
    best = max(results, key=lambda r: r.speedup)
    final = best.kernel
    spec = load_spec(problem.spec)
    evaluation = ProblemEval(
        problem=problem.id,
        family=problem.family,
        seed_us=best.seed_us,
        final_us=best.final_us,
        ab=best.ab,
        kept_seed=final == problem.seed,
        plan=[str(s) for s in best.plan],
        accepted=[str(s.stage) for s in best.stages if s.accepted],
        runs=[r.speedup for r in results],
        baselines_us=baselines(problem.id),
        correctness=held_out(runner, problem, final),
        llm=LLMUse(**{k: getattr(usage, k) for k in LLMUse.model_fields}),
    )
    try:
        evaluation.seed_kernel_us = runner.launches(problem.seed, **_timed(problem)).kernel_us
        evaluation.final_kernel_us = runner.launches(final, **_timed(problem)).kernel_us
    except WorkerError:
        pass
    if profile is not None and profile.roofline is not None:
        variant = spec.variant(TIMED_VARIANT)
        evaluation.roofline_us = roofline_us(
            spec.flop(variant), spec.bytes_moved(variant), spec.dtype(variant), profile
        )
    evaluation.wall_s = time.monotonic() - start
    return evaluation


def held_out(runner: Runner, problem: Problem, kernel: Path) -> dict[str, str]:
    """Correctness on draws optimization never saw, and on the original KernelBench shapes."""
    spec = load_spec(problem.spec)
    dtype = spec.dtype(spec.variant(TIMED_VARIANT))
    checks = {}
    for variant in ("ci", TIMED_VARIANT, HELD_OUT):
        if variant not in spec.variants:
            continue
        result = runner.check(
            kernel,
            problem.reference,
            spec=problem.spec,
            variant=variant,
            first_draw=FRESH_DRAW,
            dtype=dtype if variant == HELD_OUT else None,
        )
        if result.passed:
            checks[variant] = "pass"
        elif result.error and "out of memory" in result.error:
            checks[variant] = "skipped: out of memory"
        else:
            worst = max(result.diffs, key=lambda d: d.max_abs, default=None)
            reason = result.error.strip().splitlines()[-1][:160] if result.error else None
            if reason is None and worst is not None:
                reason = f"{worst.mismatched} mismatched, max abs {worst.max_abs:.3g}"
            checks[variant] = f"fail: {reason}"
    return checks


def baselines(problem_id: str, path: Path = RESULTS) -> dict[str, float]:
    return {
        row["mode"]: float(row["time_us"])
        for row in load_results(path)
        if row["problem"] == problem_id
        and row["variant"] == TIMED_VARIANT
        and row["mode"] in BASELINES
        and row["time_us"]
    }


def roofline_us(
    flop: float | None, moved: float | None, dtype: str, profile: DeviceProfile
) -> float | None:
    """The shortest time the measured matmul peak and DRAM read bandwidth allow."""
    roofline = profile.roofline
    limits = []
    if flop and (peak := roofline.tflops.get(dtype)):
        limits.append(flop / (peak * 1e12))
    if moved:
        limits.append(moved / (roofline.dram_read_gbps * 1e9))
    return max(limits) * 1e6 if limits else None


def save(evaluation: ProblemEval, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{evaluation.problem}.json").write_text(evaluation.model_dump_json(indent=1))


def load_run(run_dir: Path) -> tuple[RunConfig, list[ProblemEval]]:
    config = RunConfig.model_validate_json((run_dir / "config.json").read_text())
    problems = [
        ProblemEval.model_validate_json(path.read_text())
        for path in sorted(run_dir.glob("*.json"))
        if path.name not in ("config.json", "summary.json")
    ]
    return config, problems


def geomean(values: list[float]) -> float | None:
    values = [v for v in values if v and v > 0]
    return math.exp(sum(map(math.log, values)) / len(values)) if values else None


def summarize(problems: list[ProblemEval], min_effect: float = 0.02) -> dict:
    """Aggregates of one run. A regression is a final more than ``min_effect`` slower than its
    seed; a kept seed counts as neither improved nor regressed."""
    n = len(problems)
    use = LLMUse()
    for p in problems:
        for field in LLMUse.model_fields:
            setattr(use, field, getattr(use, field) + getattr(p.llm, field))
    fractions = [f for p in problems if (f := p.roofline_fraction) is not None]
    checks: dict[str, dict[str, int]] = {}
    for p in problems:
        for variant, outcome in p.correctness.items():
            kind = outcome.split(":")[0]
            checks.setdefault(variant, {}).setdefault(kind, 0)
            checks[variant][kind] += 1
    return {
        "problems": n,
        "improved": sum(1 for p in problems if not p.kept_seed and p.speedup > 1 + min_effect),
        "kept_seed": sum(1 for p in problems if p.kept_seed),
        "regressions": sum(1 for p in problems if p.speedup < 1 - min_effect),
        "incorrect": sum(1 for p in problems if not p.correct),
        "geomean_over_seed": geomean([p.speedup for p in problems]),
        "median_over_seed": statistics.median([p.speedup for p in problems]) if n else None,
        "geomean_kernel_over_seed": geomean([s for p in problems if (s := p.kernel_speedup)]),
        **{
            f"geomean_over_{b.replace('-', '_')}": geomean(
                [s for p in problems if (s := p.over(b))]
            )
            for b in BASELINES
        },
        **{
            f"faster_than_{b.replace('-', '_')}": sum(
                1 for p in problems if (s := p.over(b)) and s > 1
            )
            for b in BASELINES
        },
        "correctness": checks,
        "median_roofline_fraction": statistics.median(fractions) if fractions else None,
        "above_roofline": sorted(
            p.problem for p in problems if (p.roofline_fraction or 0) > ABOVE_ROOF
        ),
        "llm": {
            **use.model_dump(),
            "cache_hit_share": use.cache_hit_share,
            "cost_per_problem": use.cost_usd / n if n else None,
        },
        "wall_s_per_problem": sum(p.wall_s for p in problems) / n if n else None,
    }


def _timed(problem: Problem) -> dict:
    return {"spec": problem.spec, "variant": TIMED_VARIANT}


def differences(config: RunConfig, baseline: RunConfig) -> str:
    """What a run changes from the baseline run, as the ablation table's label."""
    labels = {
        "kb": "no KB",
        "arch_gating": "KB without arch gating",
        "facts": "no facts",
        "reanalysis": "no re-analysis",
        "rollback": "no rollback",
    }
    changed = [labels[k] for k in labels if getattr(config, k) != getattr(baseline, k)]
    if config.max_iters != baseline.max_iters:
        changed.append(f"T = {config.max_iters}")
    if config.best_of != baseline.best_of:
        changed.append(f"best-of-{config.best_of}")
    if (config.model, config.effort) != (baseline.model, baseline.effort):
        changed.append(f"{config.model}, effort {config.effort}")
    return ", ".join(changed) or "baseline"


def report(runs: dict[str, Path]) -> str:
    """Markdown: a summary row per run (the first is the baseline), then each run's problems."""
    loaded = {name: load_run(path) for name, path in runs.items()}
    base = next(iter(loaded.values()))[0]
    lines = [
        "| Run | Changes | Problems | Improved | Regressions | Incorrect | Over seed | Over eager "
        "| Over compile | Over max-autotune | $ per problem | Minutes per problem |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, (config, problems) in loaded.items():
        s = summarize(problems)
        lines.append(
            f"| {name} | {differences(config, base)} | {s['problems']} | {s['improved']} "
            f"| {s['regressions']} | {s['incorrect']} | {_x(s['geomean_over_seed'])} "
            f"| {_x(s['geomean_over_eager'])} | {_x(s['geomean_over_compile'])} "
            f"| {_x(s['geomean_over_max_autotune'])} | {_f(s['llm']['cost_per_problem'], 2)} "
            f"| {_f((s['wall_s_per_problem'] or 0) / 60, 1)} |"
        )
    for name, (_, problems) in loaded.items():
        lines += [
            "",
            f"### {name}",
            "",
            "| Problem | Seed | Final | Over seed | Kernel only | Over eager | Over compile "
            "| Roofline | Held-out checks | Accepted |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for p in sorted(problems, key=lambda p: p.problem):
            checks = ", ".join(f"{k} {v}" for k, v in p.correctness.items() if v != "pass")
            lines.append(
                f"| {p.problem} | {p.seed_us:.1f} us | {p.final_us:.1f} us | {p.speedup:.2f}x "
                f"| {_x(p.kernel_speedup)} | {_x(p.over('eager'))} | {_x(p.over('compile'))} "
                f"| {_f(p.roofline_fraction, 2)} | {checks or 'pass'} "
                f"| {', '.join(p.accepted) or '-'} |"
            )
    return "\n".join(lines) + "\n"


def _x(value: float | None) -> str:
    return f"{value:.2f}x" if value is not None else "-"


def _f(value: float | None, digits: int) -> str:
    return f"{value:.{digits}f}" if value is not None else "-"
