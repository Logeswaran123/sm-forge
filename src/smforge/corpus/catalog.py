"""The corpus on disk: problems, their metadata and the dev/eval split.

A problem is a directory ``corpus/<source>/<id>/`` holding ``reference.py``, ``spec.yaml`` and
``problem.yaml``, and once seeded, ``kernel.py``.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

CORPUS_DIR = Path("corpus")
SPLITS = "splits.yaml"


class Problem(BaseModel):
    id: str
    source: str
    family: str
    upstream: str | None = None
    notes: list[str] = Field(default_factory=list)
    path: Path = Field(exclude=True)

    @property
    def reference(self) -> Path:
        return self.path / "reference.py"

    @property
    def spec(self) -> Path:
        return self.path / "spec.yaml"

    @property
    def seed(self) -> Path:
        return self.path / "kernel.py"


def load_problems(root: Path = CORPUS_DIR) -> list[Problem]:
    problems = []
    for meta in sorted(root.glob("*/*/problem.yaml")):
        problems.append(Problem(**yaml.safe_load(meta.read_text()), path=meta.parent))
    return sorted(problems, key=lambda p: p.id)


def save_problem(problem: Problem) -> None:
    problem.path.mkdir(parents=True, exist_ok=True)
    data = problem.model_dump(mode="json", exclude_defaults=True)
    (problem.path / "problem.yaml").write_text(yaml.safe_dump(data, sort_keys=False))


def load_splits(root: Path = CORPUS_DIR) -> dict[str, list[str]]:
    path = root / SPLITS
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def make_splits(
    problems: list[Problem],
    existing: dict[str, list[str]],
    eval_fraction: float,
    seed: int = 0,
) -> dict[str, list[str]]:
    """Stratified by family. Problems already assigned keep their split, so eval stays frozen."""
    assigned = {pid: name for name, ids in existing.items() for pid in ids}
    by_family: dict[str, list[str]] = defaultdict(list)
    for problem in problems:
        if problem.id not in assigned:
            by_family[problem.family].append(problem.id)
    rng = random.Random(seed)
    splits = {name: list(ids) for name, ids in existing.items()}
    for family in sorted(by_family):
        ids = sorted(by_family[family])
        rng.shuffle(ids)
        count = round(len(ids) * eval_fraction)
        splits.setdefault("eval", []).extend(ids[:count])
        splits.setdefault("dev", []).extend(ids[count:])
    return {name: sorted(ids) for name, ids in sorted(splits.items())}


def save_splits(splits: dict[str, list[str]], root: Path = CORPUS_DIR) -> None:
    (root / SPLITS).write_text(yaml.safe_dump(splits, sort_keys=False))
