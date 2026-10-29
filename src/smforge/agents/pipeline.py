"""The full pipeline on one problem: analyze, plan, run each stage, re-measure after each win.

The seed is measured (timing, facts, launches, ncu) and analyzed, and the planner orders the
stages the issues route to. Each stage gets the issues routed to it. An accepted stage's output
becomes the next stage's input and is measured again; fact-backed issues are recomputed from the
new measurements and the analyzer's issues carry over. At the end the final kernel is compared
with the seed, and one that is not faster is discarded, so no regression is finalized.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

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
from smforge.harness.runner import Runner
from smforge.kb.loader import KnowledgeBase
from smforge.llm.client import BudgetExceeded, Completer, Usage
from smforge.models import DeviceProfile, Issue, Stage, Verdict
from smforge.spec import load_spec

log = logging.getLogger(__name__)


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
) -> PipelineResult:
    workdir.mkdir(parents=True, exist_ok=True)
    spec = load_spec(problem.spec)
    dtype = spec.dtype(spec.variant(TIMED_VARIANT))
    kernel = problem.seed
    measured = measure(runner, kernel, problem)
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
    stages, reason, used = plan(llm, summary, issues)
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
            measured = measure(runner, kernel, problem)
            found = pre_analyze(kernel.read_text(), measured, dtype, profile)
            issues = found + [i for i in issues if i.source == "llm"]
    if kernel != problem.seed:
        comparison = runner.compare(problem.seed, kernel, spec=problem.spec, variant=TIMED_VARIANT)
        result.seed_us, result.final_us = comparison.baseline_us, comparison.candidate_us
        if comparison.verdict is Verdict.FASTER:
            result.kernel = kernel
        else:
            log.warning(
                "%s: final kernel is %s than the seed; keeping the seed",
                problem.id,
                comparison.verdict,
            )
            result.final_us = result.seed_us
    (workdir / "output.py").write_text(result.kernel.read_text())
    (workdir / "pipeline.json").write_text(json.dumps(_record(result), indent=1))
    return result


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
        "cost_usd": round(result.usage.cost_usd, 4),
    }
