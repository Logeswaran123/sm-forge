"""Plots of eval runs: the finals on the roofline, per-problem speedups, and the ablation runs.

The roofline shows each problem with a FLOP count three times: eager PyTorch, the seed and the
final kernel, under one roof per dtype at the measured matmul peak and DRAM read bandwidth. A
final above its roof does less work than the reference's FLOP count (an algorithmic change).

    python scripts/eval_plots.py --run eval
    python scripts/eval_plots.py --ablations abl-base,abl-no-kb,abl-t5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

from smforge.config import get_config
from smforge.corpus.catalog import load_problems
from smforge.device.profile import find_profile
from smforge.evaluation import differences, load_run, summarize
from smforge.spec import load_spec

EVAL_DIR = Path("results") / "eval"
SURFACE, INK, SECONDARY, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# Categorical slots 1, 2 and 3 in fixed order: the scatter's three series, validated all-pairs.
SERIES = {"eager": ("#2a78d6", "o"), "final": ("#eb6834", "D"), "seed": ("#1baf7a", "^")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="an eval run's name: roofline and speedup plots")
    parser.add_argument("--ablations", help="comma-separated run names, the first the baseline")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.run:
        roofline(args.run, args.output_dir / f"{args.run}_roofline.png")
        speedups(args.run, args.output_dir / f"{args.run}_speedups.png")
    if args.ablations:
        ablations(args.ablations.split(","), args.output_dir / "ablations.png")


def roofline(run: str, output: Path) -> None:
    profile = find_profile(get_config().hardware_dir, get_config().device)
    if profile is None or profile.roofline is None:
        raise SystemExit("no device profile with a roofline; run `smforge device --all`")
    roof = profile.roofline
    specs = {p.id: p.spec for p in load_problems()}
    _, problems = load_run(EVAL_DIR / run)
    points: dict[str, list[tuple[float, float]]] = {name: [] for name in SERIES}
    dtypes, skipped = set(), 0
    for p in problems:
        spec = load_spec(specs[p.problem])
        variant = spec.variant("bench-gpu")
        flop, moved = spec.flop(variant), spec.bytes_moved(variant)
        if not flop or not moved:
            skipped += 1
            continue
        dtypes.add(spec.dtype(variant))
        times = {"eager": p.baselines_us.get("eager"), "seed": p.seed_us, "final": p.final_us}
        for name, time_us in times.items():
            if time_us:
                points[name].append((flop / moved, flop / time_us / 1e6))
    fig, ax = _axes((9, 6))
    xs = [x for series in points.values() for x, _ in series] or [1.0, 1000.0]
    span = [min(xs) / 2, max(xs) * 2]
    labelled: list[float] = []
    for dtype in sorted(dtypes, key=lambda d: -roof.tflops.get(d, 0)):
        peak = roof.tflops.get(dtype)
        if peak is None:
            continue
        # a roof within 15% of one already labelled gets its label below the line
        below = any(abs(peak / other - 1) < 0.15 for other in labelled)
        labelled.append(peak)
        ridge = roof.ridge(dtype)
        at = [span[0], ridge, span[1]] if span[0] < ridge < span[1] else span
        ax.plot(
            at,
            [min(peak, x * roof.dram_read_gbps / 1e3) for x in at],
            color=SECONDARY,
            linewidth=1.5,
            zorder=1,
        )
        ax.annotate(
            f"{dtype} {peak:.0f} TFLOPS",
            (span[1], peak),
            xytext=(-4, -10 if below else 4),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color=SECONDARY,
        )
    for name, (color, marker) in SERIES.items():
        if points[name]:
            ax.scatter(
                *zip(*points[name], strict=True),
                s=50,
                color=color,
                marker=marker,
                edgecolors=SURFACE,
                linewidths=2,
                label=name,
                zorder=3,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)", color=INK)
    ax.set_ylabel("Achieved TFLOPS", color=INK)
    note = f"; {skipped} without a FLOP count not shown" if skipped else ""
    ax.set_title(
        f"{run}: {len(problems)} problems on {profile.device.name}{note}",
        color=INK,
        fontsize=11,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    _save(fig, output)


def speedups(run: str, output: Path) -> None:
    _, problems = load_run(EVAL_DIR / run)
    ranked = sorted(problems, key=lambda p: p.speedup, reverse=True)
    fig, ax = _axes((max(8, len(ranked) * 0.28), 6))
    color = SERIES["final"][0]
    ax.bar(
        range(len(ranked)),
        [p.speedup for p in ranked],
        width=0.8,
        color=color,
        edgecolor=SURFACE,
        linewidth=2,
        zorder=2,
    )
    ax.axhline(1, color=SECONDARY, linewidth=1, zorder=3)
    values = [p.speedup for p in ranked]
    if values and max(values) / min(values) >= 3:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=()))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.set_xticks(
        range(len(ranked)), [p.problem for p in ranked], rotation=90, fontsize=7, color=SECONDARY
    )
    ax.set_ylabel("Final over seed", color=INK)
    ax.set_title(
        f"{run}: final kernel over its seed, per problem", color=INK, fontsize=11, loc="left"
    )
    _save(fig, output)


def ablations(runs: list[str], output: Path) -> None:
    loaded = [(name, *load_run(EVAL_DIR / name)) for name in runs]
    base = loaded[0][1]
    rows = []
    for _name, config, problems in loaded:
        s = summarize(problems)
        rows.append((differences(config, base), s["geomean_over_seed"] or 0, s["regressions"]))
    fig, ax = _axes((8, 0.45 * len(rows) + 1.5))
    color = SERIES["eager"][0]
    labels = [label for label, *_ in rows]
    values = [value for _, value, _ in rows]
    ax.barh(
        range(len(rows)), values, height=0.7, color=color, edgecolor=SURFACE, linewidth=2, zorder=2
    )
    ax.axvline(values[0], color=SECONDARY, linewidth=1, linestyle="--", zorder=3)
    for i, (_, value, regressions) in enumerate(rows):
        text = f"{value:.2f}x" + (f", {regressions} regressions" if regressions else "")
        ax.annotate(
            text,
            (value, i),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK,
        )
    ax.set_yticks(range(len(rows)), labels, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Geomean final over seed (x)", color=INK)
    ax.set_title(
        "Ablations: the same problems under each config (dashed: baseline)",
        color=INK,
        fontsize=11,
        loc="left",
    )
    ax.set_xlim(0, max(values) * 1.25)
    _save(fig, output)


def _axes(size: tuple[float, float]):
    fig, ax = plt.subplots(figsize=size, dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    return fig, ax


def _save(fig, output: Path) -> None:
    fig.tight_layout()
    fig.savefig(output, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
