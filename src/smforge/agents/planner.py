"""Choosing and ordering the stages for one kernel.

Every issue type routes to one stage. The planner LLM picks and orders stages among those the
issues route to; stages with a fact-backed issue are added after its choices, and at most
``max_stages`` stages besides autotuning are kept, in that priority. The order is then corrected
for hard dependencies (autotuning after everything else, fusion after
algorithmic and dtype changes, device tuning after fusion and memory work). If the LLM call
fails, the routed stages run in their canonical order.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel

from smforge.agents.analyze import issues_text
from smforge.llm.client import BudgetExceeded, Completer, Usage
from smforge.llm.structured import ask
from smforge.models import Issue, IssueType, Stage

log = logging.getLogger(__name__)
PROMPTS = Path(__file__).parent.parent / "llm" / "prompts"

ROUTES = {
    IssueType.ALGEBRAIC_SIMPLIFICATION: Stage.ALGORITHMIC,
    IssueType.NO_TENSOR_CORE_MMA: Stage.ALGORITHMIC,
    IssueType.TF32_PRECISION_RISK: Stage.DTYPE_FIX,
    IssueType.UPCAST_OPERANDS: Stage.DTYPE_FIX,
    IssueType.UNFUSED_OPS: Stage.FUSION,
    IssueType.LAUNCH_BOUND: Stage.FUSION,
    IssueType.HOST_SYNC_IN_FORWARD: Stage.FUSION,
    IssueType.UNCOALESCED_ACCESS: Stage.MEMORY_ACCESS,
    IssueType.BANK_CONFLICT: Stage.MEMORY_ACCESS,
    IssueType.PER_CALL_WEIGHT_PREP: Stage.MEMORY_ACCESS,
    IssueType.MISSING_PIPELINING: Stage.TENSOR_DESCRIPTORS,
    IssueType.GRID_UNDERFILL: Stage.PERSISTENT_KERNEL,
    IssueType.TAIL_WAVE: Stage.PERSISTENT_KERNEL,
    IssueType.SPLIT_K_OPPORTUNITY: Stage.PERSISTENT_KERNEL,
    IssueType.REGISTER_SPILL: Stage.DEVICE_SPECIFIC,
    IssueType.LOW_OCCUPANCY: Stage.DEVICE_SPECIFIC,
    IssueType.SMEM_OVER_BUDGET: Stage.DEVICE_SPECIFIC,
    IssueType.OPEN_ENDED: Stage.OPEN_ENDED,
    IssueType.UNTUNED_CONFIG: Stage.AUTOTUNING,
}
AFTER = {
    Stage.FUSION: {Stage.ALGORITHMIC, Stage.DTYPE_FIX},
    Stage.DEVICE_SPECIFIC: {
        Stage.ALGORITHMIC,
        Stage.FUSION,
        Stage.MEMORY_ACCESS,
        Stage.PERSISTENT_KERNEL,
    },
    Stage.OPEN_ENDED: {Stage.ALGORITHMIC},
    Stage.AUTOTUNING: set(Stage) - {Stage.AUTOTUNING},
}
"""A stage runs after each of these that is in the plan."""


class PlanReply(BaseModel):
    stages: list[Stage]
    reason: str


def routed(issues: list[Issue]) -> list[Stage]:
    """The stages the issues route to, in canonical order."""
    stages = {ROUTES[issue.type] for issue in issues}
    return [stage for stage in Stage if stage in stages]


def order(stages: list[Stage]) -> list[Stage]:
    """``stages`` in the given order where possible, each after its hard dependencies."""
    remaining, ordered = list(dict.fromkeys(stages)), []
    while remaining:
        ready = next(s for s in remaining if not (AFTER.get(s, set()) & set(remaining) - {s}))
        ordered.append(ready)
        remaining.remove(ready)
    return ordered


def plan(
    llm: Completer, summary: str, issues: list[Issue], max_stages: int | None = None
) -> tuple[list[Stage], str, Usage]:
    candidates = routed(issues)
    usage = Usage()
    if len(candidates) < 2:
        return candidates, "one routed stage", usage
    by_stage = "\n".join(
        f"- {stage}: " + ", ".join(sorted({i.type for i in issues if ROUTES[i.type] is stage}))
        for stage in candidates
    )
    task = (
        f"Analyzer summary:\n{summary}\n\nIssues:\n{issues_text(issues)}\n\n"
        f"Stages these issues route to:\n{by_stage}"
    )
    if max_stages is not None:
        task += f"\n\nChoose at most {max_stages} stages besides autotuning."
    try:
        reply, raw = ask(
            llm,
            (PROMPTS / "plan.md").read_text(),
            [{"role": "user", "content": task}],
            PlanReply,
            "planner",
        )
    except BudgetExceeded:
        raise
    except Exception as e:
        log.warning("planner failed, using the canonical order: %s", e)
        return _limit(candidates, max_stages), f"fallback: {e}", usage
    usage.add(raw.usage)
    chosen = [stage for stage in reply.stages if stage in candidates]
    required = {ROUTES[i.type] for i in issues if i.source != "llm"}
    chosen += [stage for stage in candidates if stage in required and stage not in chosen]
    return order(_limit(chosen, max_stages)), reply.reason, usage


def _limit(stages: list[Stage], max_stages: int | None) -> list[Stage]:
    """The first ``max_stages`` stages other than autotuning, and autotuning if present."""
    if max_stages is None:
        return stages
    others = [stage for stage in stages if stage is not Stage.AUTOTUNING][:max_stages]
    return others + [Stage.AUTOTUNING] * (Stage.AUTOTUNING in stages)
