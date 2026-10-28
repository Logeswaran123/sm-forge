"""Result tables: ``results/baselines.csv``, one row per problem, variant and mode, and
``results/stages.csv``, one row per problem and optimization stage."""

from __future__ import annotations

import csv
import subprocess
from datetime import date
from pathlib import Path

RESULTS = Path("results") / "baselines.csv"
FIELDS = (
    "problem",
    "variant",
    "mode",
    "dtype",
    "time_us",
    "tflops",
    "gbps",
    "gpu",
    "driver",
    "torch",
    "triton",
    "commit",
    "date",
)
KEY = ("problem", "variant", "mode")
STAGES = Path("results") / "stages.csv"
STAGE_FIELDS = (
    "problem",
    "stage",
    "accepted",
    "input_us",
    "output_us",
    "speedup",
    "attempts",
    "outcomes",
    "cost_usd",
    "model",
    "gpu",
    "triton",
    "commit",
    "date",
)
STAGE_KEY = ("problem", "stage")


def load_results(path: Path = RESULTS) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def upsert(
    rows: list[dict],
    path: Path = RESULTS,
    fields: tuple[str, ...] = FIELDS,
    key: tuple[str, ...] = KEY,
) -> None:
    """Replace rows with the same key; keep the rest."""
    merged = {tuple(row[k] for k in key): row for row in load_results(path)}
    for row in rows:
        merged[tuple(str(row[k]) for k in key)] = {f: row.get(f, "") for f in fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged[key] for key in sorted(merged))


def provenance(device: dict) -> dict[str, str]:
    """The columns that say where and with what a timing was taken."""
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    return {
        "gpu": device["name"],
        "driver": device.get("driver") or "",
        "torch": device["torch"],
        "triton": device["triton"],
        "commit": commit,
        "date": date.today().isoformat(),
    }
