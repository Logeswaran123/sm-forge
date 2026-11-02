"""The full pipeline on one problem: analyze, plan, run each stage, re-measure after each win.

The seed is measured (timing, facts, launches, ncu) and analyzed, and the planner orders the
stages the issues route to. Each stage gets the issues routed to it. An accepted stage's output
becomes the next stage's input and is measured again; fact-backed issues are recomputed from the
new measurements and the analyzer's issues carry over. At the end the last accepted kernel runs
under compute-sanitizer memcheck on ``ci`` and is compared with the seed in several fresh
workers; one that fails either gives way to the stage output before it, and the seed is kept when
none passes, so no regression is finalized.

``Options`` turns parts of this off for the ablations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from smforge import lessons
from smforge.agents.analyze import (
    TIMED_VARIANT,
    Measurements,
    analyze,
    issues_text,
    measure,
    pre_analyze,
)
from smforge.agents.candidate import confirm
from smforge.agents.optimize import StageResult, optimize_stage
from smforge.agents.planner import ROUTES, plan
from smforge.corpus.catalog import Problem
from smforge.harness.runner import Runner
from smforge.kb.loader import KnowledgeBase
from smforge.llm.client import BudgetExceeded, Completer, Usage
from smforge.models import DeviceProfile, Issue, SanitizerReport, Stage
from smforge.spec import load_spec

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Options:
    """What the pipeline does, all on by default. Each off is one ablation.

    ``facts``: per-kernel facts, launches and counters in the prompts, the fact-backed issues,
    and fact deltas in feedback (off: the LLM sees the timing and the code). ``reanalysis``:
    measure again and recompute the fact-backed issues after an accepted stage (off: every stage
    works from the seed's measurements). ``rollback``: keep a stage's output only if it beats the
    stage input, and never finalize a kernel slower than the seed (off: a stage keeps its fastest
    correct candidate, and the last stage's output is final if memcheck passes).
    """

    facts: bool = True
    reanalysis: bool = True
    rollback: bool = True


@dataclass
class PipelineResult:
    problem: str
    summary: str
    plan: list[Stage]
    plan_reason: str
    issues: list[Issue]
    stages: list[StageResult]
    seed_us: float
    final_us: float
    kernel: Path
    usage: Usage = field(default_factory=Usage)
    memcheck: list[tuple[Stage, SanitizerReport]] = field(default_factory=list)
    ab: list[float] = field(default_factory=list)

    @property
    def speedup(self) -> float:
        return self.seed_us / self.final_us


def run_pipeline(
    problem: Problem,
    runner: Runner,
    llm: Completer,
    kb: KnowledgeBase,
    profile: DeviceProfile | None,
    max_iters: int,
    workdir: Path,
    keep: int | None = 2,
    max_stages: int | None = None,
    ncu: bool = True,
    memcheck: bool = True,
    ledger: Path | None = None,
    options: Options | None = None,
) -> PipelineResult:
    """``ledger``, the problem's lessons file on this GPU, is read into each stage's prompt
    and gets a line for every candidate."""
    options = options or Options()
    workdir.mkdir(parents=True, exist_ok=True)
    past = lessons.read(ledger)
    if ledger is not None:
        lessons.start(ledger, "pipeline")
    spec = load_spec(problem.spec)
    dtype = spec.dtype(spec.variant(TIMED_VARIANT))
    kernel = problem.seed
    measured = _measure(runner, kernel, problem, ncu, options)
    found = pre_analyze(kernel.read_text(), measured, dtype, profile) if options.facts else []
    usage = Usage()
    try:
        summary, issues, used = analyze(llm, problem, kernel.read_text(), measured, found, profile)
        usage.add(used)
    except BudgetExceeded:
        raise
    except Exception as e:
        log.warning("%s: analyzer failed, using fact-backed issues only: %s", problem.id, e)
        summary, issues = f"analyzer failed: {e}", list(found)
    stages, reason, used = plan(llm, summary, issues, max_stages)
    usage.add(used)
    result = PipelineResult(
        problem.id,
        summary,
        stages,
        reason,
        list(issues),
        [],
        measured.timing.time_us,
        measured.timing.time_us,
        kernel,
        usage,
    )
    for stage in stages:
        notes = f"Analyzer summary: {summary}\n\nIssues for this stage:\n" + issues_text(
            [i for i in issues if ROUTES[i.type] is stage]
        )
        if past:
            notes += f"\n\nEarlier runs on this problem and GPU (lessons ledger):\n{past}"
        stage_result = optimize_stage(
            problem,
            stage,
            kernel,
            runner,
            llm,
            kb,
            profile,
            max_iters,
            workdir / stage.value,
            keep,
            notes,
            measured,
            memcheck,
            options.rollback,
        )
        result.stages.append(stage_result)
        result.usage.add(stage_result.usage)
        if ledger is not None:
            for a in stage_result.attempts:
                lessons.add(
                    ledger,
                    f"{stage} attempt {a.attempt}: {a.outcome}"
                    f"{lessons.speedup(a.speedup, 'its input')}: {a.summary or a.detail}",
                )
        if stage_result.accepted:
            kernel = stage_result.kernel
            if options.reanalysis:
                measured = _measure(runner, kernel, problem, ncu, options)
                if options.facts:
                    found = pre_analyze(kernel.read_text(), measured, dtype, profile)
                issues = found + [i for i in issues if i.source == "llm"]
    finalize(result, problem, runner, memcheck, options.rollback)
    if ledger is not None:
        kept = "the seed" if result.kernel == problem.seed else "the last clean accepted kernel"
        lessons.add(ledger, f"final: {result.speedup:.2f}x over the seed, kept {kept}")
    (workdir / "output.py").write_text(result.kernel.read_text())
    (workdir / "pipeline.json").write_text(json.dumps(_record(result), indent=1))
    return result


def finalize(
    result: PipelineResult, problem: Problem, runner: Runner, memcheck: bool, rollback: bool = True
) -> None:
    """Keep the last accepted kernel that passes memcheck and beats the seed in every A/B run.

    Accepted kernels are tried from the last stage back (``confirm``: memcheck on ``ci``, then
    the A/B in fresh workers, the slowest counting). The seed is kept when none passes. Without
    ``rollback`` a kernel that passes memcheck is kept however its A/B runs come out.
    """
    for stage in reversed([s for s in result.stages if s.accepted]):
        check = confirm(runner, problem, problem.seed, stage.kernel, memcheck)
        if check.memcheck is not None:
            result.memcheck.append((stage.stage, check.memcheck))
        if check.comparisons:
            result.seed_us = check.worst.baseline_us
        clean = check.memcheck is None or check.memcheck.ok
        if check.ok or (not rollback and clean and check.comparisons):
            result.kernel, result.final_us = stage.kernel, check.worst.candidate_us
            result.ab = [c.speedup for c in check.comparisons]
            if check.unstable:
                log.warning(
                    "%s: the %s output's A/B runs spread %.1f%%, beyond the A/A band; the "
                    "slowest counts",
                    problem.id,
                    stage.stage,
                    100 * check.spread,
                )
            return
        log.warning("%s: the %s output is not kept: %s", problem.id, stage.stage, check.reason())
    result.final_us = result.seed_us


def _measure(
    runner: Runner, kernel: Path, problem: Problem, ncu: bool, options: Options
) -> Measurements:
    measured = measure(runner, kernel, problem, ncu and options.facts)
    measured.shown = options.facts
    return measured


def memcheck_summary(result: PipelineResult) -> str:
    """``stage=ok`` or ``stage=N errors`` for each kernel memcheck ran on, in order."""
    return ",".join(
        f"{stage}=" + ("ok" if r.ok else f"{r.errors} errors" if r.errors else "failed")
        for stage, r in result.memcheck
    )


def _record(result: PipelineResult) -> dict:
    return {
        "problem": result.problem,
        "summary": result.summary,
        "plan": [str(s) for s in result.plan],
        "plan_reason": result.plan_reason,
        "issues": [i.model_dump(mode="json") for i in result.issues],
        "stages": [
            {
                "stage": str(s.stage),
                "accepted": s.accepted,
                "speedup": round(s.speedup, 4),
                "outcomes": [a.outcome for a in s.attempts],
                "cost_usd": round(s.usage.cost_usd, 4),
            }
            for s in result.stages
        ],
        "seed_us": result.seed_us,
        "final_us": result.final_us,
        "ab": [round(x, 4) for x in result.ab],
        "memcheck": [
            {"stage": str(stage), **report.model_dump(mode="json")}
            for stage, report in result.memcheck
        ],
        "cost_usd": round(result.usage.cost_usd, 4),
    }
