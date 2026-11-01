"""Loading the knowledge base, filtering it for a device, and rendering one stage's slice."""

from __future__ import annotations

import difflib
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from smforge.kb.schema import Constraint, Example, Measurement, Pattern
from smforge.models import DeviceProfile, Stage

KB_DIR = Path("knowledge_base") / "triton"
MEASURED = "measured.yaml"
TOKEN_BUDGET = 6000
CHARS_PER_TOKEN = 4


class KnowledgeBaseError(ValueError):
    pass


@dataclass
class KnowledgeBase:
    root: Path
    constraints: list[Constraint] = field(default_factory=list)
    patterns: list[Pattern] = field(default_factory=list)
    examples: dict[str, Example] = field(default_factory=dict)

    def check_files(self, constraint: Constraint) -> tuple[Path, Path]:
        """``wrong.py`` and ``correct.py`` of a constraint."""
        path = self.root / "checks" / constraint.id
        return path / "wrong.py", path / "correct.py"

    def example_files(self, example: Example) -> tuple[Path, Path]:
        """``before.py`` and ``after.py`` of an example."""
        path = self.root / "examples" / example.id
        return path / "before.py", path / "after.py"

    def for_device(self, profile: DeviceProfile) -> KnowledgeBase:
        """Entries whose arch range admits the device and whose required probes pass on it."""
        major, minor = profile.device.capability
        sm, features = 10 * major + minor, profile.features

        def fits(entry: Constraint | Pattern) -> bool:
            return entry.arch.admits(sm) and all(features.get(name) for name in entry.requires)

        return replace(
            self,
            constraints=[c for c in self.constraints if fits(c)],
            patterns=[p for p in self.patterns if fits(p)],
        )

    def render(self, stage: Stage, families: Iterable[str] = (), budget: int = TOKEN_BUDGET) -> str:
        """The stage's constraints and patterns as Markdown, within ``budget`` tokens.

        Critical constraints come first, then patterns for ``families``, then patterns for any
        family. Each kept pattern's example follows as a diff while the budget allows.
        """
        families = set(families)
        constraints = sorted(
            (c for c in self.constraints if not c.stages or stage in c.stages),
            key=lambda c: c.severity != "critical",
        )
        patterns = [
            p
            for p in self.patterns
            if p.stage is stage
            and (not families or "any" in p.applies_to or families & set(p.applies_to))
        ]
        patterns.sort(key=lambda p: not families & set(p.applies_to))
        left = budget * CHARS_PER_TOKEN
        kept_constraints = []
        for constraint in constraints:
            text = _constraint(constraint)
            if len(text) <= left:
                kept_constraints.append(text)
                left -= len(text)
        kept_patterns: list[tuple[Pattern, str]] = []
        for pattern in patterns:
            text = _pattern(pattern)
            if len(text) <= left:
                kept_patterns.append((pattern, text))
                left -= len(text)
        diffs = {}
        for pattern, _ in kept_patterns:
            diff = self._diff(self.examples[pattern.example])
            if len(diff) <= left:
                diffs[pattern.id] = diff
                left -= len(diff)
        sections = []
        if kept_constraints:
            sections.append("## Constraints\n\n" + "\n".join(kept_constraints))
        if kept_patterns:
            body = "\n".join(text + diffs.get(p.id, "") for p, text in kept_patterns)
            sections.append(f"## Patterns: {stage}\n\n{body}")
        return "\n".join(sections)

    def render_all(self, families: Iterable[str] = ()) -> str:
        """Every constraint, then every stage's patterns with their examples, without a budget.

        Patterns for ``families`` (and for any family) come first within each stage; patterns
        for other families are listed too, since a workspace reader can skip them.
        """
        families = set(families)
        constraints = sorted(self.constraints, key=lambda c: c.severity != "critical")
        sections = ["## Constraints\n\n" + "\n".join(_constraint(c) for c in constraints)]
        for stage in Stage:
            patterns = [p for p in self.patterns if p.stage is stage]
            if not patterns:
                continue
            patterns.sort(key=lambda p: not ({"any", *families} & set(p.applies_to)))
            body = "\n".join(_pattern(p) + self._diff(self.examples[p.example]) for p in patterns)
            sections.append(f"## Patterns: {stage}\n\n{body}")
        return "\n".join(sections)

    def _diff(self, example: Example) -> str:
        before, after = self.example_files(example)
        lines = difflib.unified_diff(
            before.read_text().splitlines(),
            after.read_text().splitlines(),
            "before.py",
            "after.py",
            n=2,
            lineterm="",
        )
        return f"Example on {example.problem}:\n```diff\n" + "\n".join(lines) + "\n```\n"


def _constraint(c: Constraint) -> str:
    return (
        f"- **{c.id}** ({c.severity}): {c.rule.strip()}\n"
        f"  Wrong:\n  ```python\n{_indent(c.wrong)}\n  ```\n"
        f"  Correct:\n  ```python\n{_indent(c.correct)}\n  ```\n"
    )


def _pattern(p: Pattern) -> str:
    families = ", ".join(p.applies_to)
    return f"### {p.id} ({families})\n\n{p.summary.strip()}\n\n{p.how.strip()}\n\n"


def _indent(code: str) -> str:
    return "\n".join(f"  {line}" if line else "" for line in code.strip().splitlines())


def load_kb(root: Path = KB_DIR) -> KnowledgeBase:
    kb = KnowledgeBase(root)
    constraints = root / "constraints.yaml"
    if constraints.exists():
        kb.constraints = [Constraint.model_validate(c) for c in _read(constraints)["constraints"]]
    errors = []
    for path in sorted((root / "patterns").glob("*.yaml")):
        for entry in _read(path)["patterns"]:
            pattern = Pattern.model_validate(entry)
            if pattern.stage != path.stem:
                errors.append(f"{pattern.id}: stage {pattern.stage} in {path.name}")
            kb.patterns.append(pattern)
    index = root / "examples" / "index.yaml"
    if index.exists():
        for entry in _read(index)["examples"]:
            example = Example.model_validate(entry)
            kb.examples[example.id] = example
    measured = root / "examples" / MEASURED
    if measured.exists():
        for example_id, entries in _read(measured).items():
            if example_id in kb.examples:
                kb.examples[example_id].measured = [Measurement.model_validate(m) for m in entries]
    errors += _problems(kb)
    if errors:
        raise KnowledgeBaseError("invalid knowledge base:\n  " + "\n  ".join(errors))
    return kb


def record_measurement(kb: KnowledgeBase, example_id: str, measurement: Measurement) -> None:
    """Replace the example's measurement for the same GPU and variant in ``measured.yaml``."""
    path = kb.root / "examples" / MEASURED
    data = _read(path) if path.exists() else {}
    entries = [
        m
        for m in data.get(example_id, [])
        if (m["gpu"], m["variant"]) != (measurement.gpu, measurement.variant)
    ]
    data[example_id] = [*entries, measurement.model_dump()]
    path.write_text(yaml.safe_dump(dict(sorted(data.items())), sort_keys=False))
    kb.examples[example_id].measured = [Measurement.model_validate(m) for m in data[example_id]]


def _problems(kb: KnowledgeBase) -> list[str]:
    errors = []
    ids = [c.id for c in kb.constraints] + [p.id for p in kb.patterns]
    errors += [f"{i}: duplicate id" for i in sorted({i for i in ids if ids.count(i) > 1})]
    for constraint in kb.constraints:
        errors += [
            f"{constraint.id}: missing {p}" for p in kb.check_files(constraint) if not p.exists()
        ]
    for pattern in kb.patterns:
        if pattern.example not in kb.examples:
            errors.append(f"{pattern.id}: unknown example {pattern.example}")
    for example in kb.examples.values():
        errors += [
            f"{example.id}: missing {p}" for p in kb.example_files(example) if not p.exists()
        ]
    used = {pattern.example for pattern in kb.patterns}
    errors += [f"{e}: no pattern uses this example" for e in sorted(set(kb.examples) - used)]
    return errors


def _read(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}
