"""Lessons ledger: what was tried on a problem and what it measured, one markdown file per GPU.

``<cache>/lessons/<gpu>/<problem>.md`` gets a section per run, with a line for every pipeline
candidate and every trial. Later runs on the same GPU read it: the pipeline adds it to each
stage's prompt, and the Claude Code workspace includes it. Timings from another GPU would
mislead, so each GPU has its own directory.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

PROMPT_LINES = 40


def ledger_path(cache_dir: Path, gpu: str, problem: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", gpu.lower()).strip("_")
    return cache_dir / "lessons" / slug / f"{problem}.md"


def start(path: Path, source: str) -> None:
    """Open a section for one run: the engine and when it started."""
    path.parent.mkdir(parents=True, exist_ok=True)
    first = not path.exists()
    with path.open("a") as f:
        if first:
            f.write(f"# {path.stem}\n")
        f.write(f"\n## {datetime.now():%Y-%m-%d %H:%M} {source}\n\n")


def add(path: Path, line: str) -> None:
    with path.open("a") as f:
        f.write(f"- {' '.join(line.split())}\n")


def read(path: Path | None, lines: int = PROMPT_LINES) -> str | None:
    """The last ``lines`` entries with their section headings, or None when there are none."""
    if path is None or not path.exists():
        return None
    kept: list[str] = []
    entries = 0
    for line in reversed(path.read_text().splitlines()):
        if line.startswith("- "):
            if entries == lines:
                continue
            entries += 1
            kept.append(line)
        elif line.startswith("## ") and kept and not kept[-1].startswith("## "):
            kept.append(line)
    return "\n".join(reversed(kept)) if entries else None


def speedup(value: float | None, over: str) -> str:
    return f" {value:.2f}x over {over}" if value is not None else ""
