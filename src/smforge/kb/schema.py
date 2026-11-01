"""Knowledge base entries and the kernel pairs that verify them.

A constraint is a rule a kernel must follow. Its pair, under ``checks/<id>/``, holds a
``wrong.py`` that fails the way ``check.expect`` says and a ``correct.py`` that passes.

A pattern is an optimization. Its example, under ``examples/<id>/``, holds a ``before.py`` and an
``after.py`` on a dev-split corpus problem; the after kernel must be correct and measurably
faster. Measurements are written to ``examples/measured.yaml`` by ``smforge kb verify --write``,
never by hand, and are not shown in prompts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from smforge.models import Stage


class Arch(BaseModel):
    """Compute capabilities an entry holds for, as ``10 * major + minor`` (sm_120 is 120)."""

    min: int = 80
    max: int | None = None

    def admits(self, sm: int) -> bool:
        return sm >= self.min and (self.max is None or sm <= self.max)


class Check(BaseModel):
    """How ``wrong.py`` must fail.

    ``static``: the static validator or the source audit reports ``match``. ``error``: the
    correctness check fails with an error containing ``match``. ``mismatch``: it runs and its
    outputs differ from the reference. ``slower``: ``correct.py`` beats it in an A/B comparison.
    """

    problem: str
    variant: str = "ci"
    expect: Literal["static", "error", "mismatch", "slower"]
    match: str | None = None


class Constraint(BaseModel):
    id: str
    severity: Literal["critical", "warning"]
    stages: list[Stage] = Field(default_factory=list, description="empty means every stage")
    arch: Arch = Field(default_factory=Arch)
    requires: list[str] = Field(default_factory=list, description="device probes that must pass")
    rule: str
    wrong: str
    correct: str
    check: Check


class Pattern(BaseModel):
    id: str
    stage: Stage
    arch: Arch = Field(default_factory=Arch)
    requires: list[str] = Field(default_factory=list, description="device probes that must pass")
    applies_to: list[str] = Field(description="corpus families, or 'any'")
    summary: str
    how: str
    example: str


class Measurement(BaseModel):
    gpu: str
    variant: str
    before_us: float
    after_us: float
    speedup: float
    triton: str
    cuda: str | None = None
    date: str


class Example(BaseModel):
    id: str
    problem: str
    variant: str = "bench-gpu"
    description: str
    measured: list[Measurement] = Field(default_factory=list)
