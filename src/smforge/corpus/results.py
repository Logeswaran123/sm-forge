"""Baseline and seed timings: ``results/baselines.csv``, one row per problem, variant and mode."""

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


def load_results(path: Path = RESULTS) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def upsert(rows: list[dict], path: Path = RESULTS) -> None:
    """Replace rows with the same problem, variant and mode; keep the rest."""
    merged = {tuple(row[k] for k in KEY): row for row in load_results(path)}
    for row in rows:
        merged[tuple(str(row[k]) for k in KEY)] = {f: row.get(f, "") for f in FIELDS}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
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
