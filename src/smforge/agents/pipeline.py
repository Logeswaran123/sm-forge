"""The full pipeline on one problem: analyze, plan, run each stage, re-measure after each win.

The seed is measured (timing, facts, launches, ncu) and analyzed, and the planner orders the
stages the issues route to. Each stage gets the issues routed to it. An accepted stage's output
becomes the next stage's input and is measured again; fact-backed issues are recomputed from the
new measurements and the analyzer's issues carry over. At the end the last accepted kernel runs
under compute-sanitizer memcheck on ``ci``; one with errors gives way to the stage output before
it. The first clean one is compared with the seed and kept only if faster, so no regression is
finalized.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from smforge import render
from smforge.agents.analyze import (
    TIMED_VARIANT,
    analyze,
    issues_text,
    measure,
    pre_analyze,
)
from smforge.agents.optimize import StageResult, optimize_stage
from smforge.agents.planner import ROUTES, plan
from smforge.corpus.catalog import Problem
from smforge.device.tools import tool_status
from smforge.harness.runner import Runner
from smforge.harness.sanitizer import SanitizerError, SanitizerUnavailable, sanitize
from smforge.kb.loader import KnowledgeBase
from smforge.llm.client import BudgetExceeded, Completer, Usage
from smforge.models import DeviceProfile, Issue, SanitizerReport, Stage, Verdict
from smforge.spec import load_spec

log = logging.getLogger(__name__)
SANITIZED_VARIANT = "ci"


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
) -> PipelineResult:
    workdir.mkdir(parents=True, exist_ok=True)
    spec = load_spec(problem.spec)
    dtype = spec.dtype(spec.variant(TIMED_VARIANT))
    kernel = problem.seed
    measured = measure(runner, kernel, problem, ncu)
    found = pre_analyze(kernel.read_text(), measured, dtype, profile)
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
        )
        result.stages.append(stage_result)
        result.usage.add(stage_result.usage)
        if stage_result.accepted:
            kernel = stage_result.kernel
            measured = measure(runner, kernel, problem, ncu)
            found = pre_analyze(kernel.read_text(), measured, dtype, profile)
            issues = found + [i for i in issues if i.source == "llm"]
    finalize(result, problem, runner, memcheck)
    (workdir / "output.py").write_text(result.kernel.read_text())
    (workdir / "pipeline.json").write_text(json.dumps(_record(result), indent=1))
    return result


def finalize(result: PipelineResult, problem: Problem, runner: Runner, memcheck: bool) -> None:
    """Keep the last accepted kernel if it is memcheck-clean and faster than the seed.

    A kernel memcheck flags is passed over for the stage output before it. The first clean one
    is compared with the seed, and the seed is kept unless it is faster.
    """
    spec = load_spec(problem.spec)
    variant = SANITIZED_VARIANT if SANITIZED_VARIANT in spec.variants else TIMED_VARIANT
    for stage in reversed([s for s in result.stages if s.accepted]):
        if memcheck:
            try:
                report = sanitize(stage.kernel, "memcheck", problem.spec, variant)
            except SanitizerError as e:
                report = SanitizerReport(tool="memcheck", ok=False, errors=0, detail=str(e))
            if report.detail is not None:
                status = tool_status("compute-sanitizer")
                if not status.ok:
                    raise SanitizerUnavailable(
                        f"compute-sanitizer stopped working: {status.detail}"
                    )
            result.memcheck.append((stage.stage, report))
            if not report.ok:
                log.warning("%s: memcheck: %s", problem.id, render.sanitizer(report))
                continue
        comparison = runner.compare(
            problem.seed, stage.kernel, spec=problem.spec, variant=TIMED_VARIANT
        )
        result.seed_us = comparison.baseline_us
        if comparison.verdict is Verdict.FASTER:
            result.kernel, result.final_us = stage.kernel, comparison.candidate_us
        else:
            log.warning(
                "%s: final kernel is %s than the seed; keeping the seed",
                problem.id,
                comparison.verdict,
            )
            result.final_us = result.seed_us
        return
    result.final_us = result.seed_us


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
        "memcheck": [
            {"stage": str(stage), **report.model_dump(mode="json")}
            for stage, report in result.memcheck
        ],
        "cost_usd": round(result.usage.cost_usd, 4),
    }
