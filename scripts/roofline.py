"""Roofline of the corpus timings: achieved TFLOPS against arithmetic intensity.

Ceilings come from the device profile (`smforge device --all`): one flat roof per dtype at the
measured matmul peak, under the measured DRAM read bandwidth. Problems without a FLOP count cannot
sit on this plot; they are listed with their bandwidth as a fraction of DRAM read instead. A problem
whose bytes fit in L2 can exceed that: the L2 flush before each sample evicts its inputs, but its
writes can stay in L2 inside the timed window.

    python scripts/roofline.py [--results results/baselines.csv] [--output plots/roofline.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from smforge.config import get_config
from smforge.corpus.catalog import load_problems
from smforge.corpus.results import RESULTS, load_results
from smforge.device.profile import find_profile
from smforge.spec import load_spec

MODES = {"eager": ("#2a78d6", "o"), "compile": ("#eb6834", "s"), "seed": ("#1baf7a", "^")}
SURFACE, INK, SECONDARY, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=Path("plots/roofline.png"))
    parser.add_argument("--variant", default="bench-gpu")
    args = parser.parse_args()

    profile = find_profile(get_config().hardware_dir, get_config().device)
    if profile is None or profile.roofline is None:
        raise SystemExit("no device profile with a roofline; run `smforge device --all`")
    roofline = profile.roofline
    problems = {p.id: p for p in load_problems()}
    points, bandwidth = [], []
    for row in load_results(args.results):
        if row["variant"] != args.variant or row["problem"] not in problems:
            continue
        spec = load_spec(problems[row["problem"]].spec)
        variant = spec.variant(args.variant)
        flop, moved = spec.flop(variant), spec.bytes_moved(variant)
        if flop and moved and row["tflops"]:
            points.append((row["mode"], flop / moved, float(row["tflops"]), row["dtype"]))
        elif row["gbps"]:
            fits = moved is not None and moved <= profile.device.l2_bytes
            bandwidth.append((row["problem"], row["mode"], float(row["gbps"]), fits))

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    lows = [x for _, x, _, _ in points] or [1.0]
    highs = [x for _, x, _, _ in points] or [1000.0]
    xs = [min(lows) / 2, max(highs) * 2]
    dtypes = sorted({d for *_, d in points}) or list(roofline.tflops)
    for dtype in dtypes:
        peak = roofline.tflops.get(dtype)
        if peak is None:
            continue
        ridge = roofline.ridge(dtype)
        roof = [xs[0], ridge, xs[1]] if xs[0] < ridge < xs[1] else xs
        ax.plot(
            roof,
            [min(peak, x * roofline.dram_read_gbps / 1e3) for x in roof],
            color=SECONDARY,
            linewidth=1.5,
            zorder=1,
        )
        ax.annotate(
            f"{dtype} {peak:.0f} TFLOPS",
            (xs[1], peak),
            xytext=(-4, 4),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color=SECONDARY,
        )
    for mode, (color, marker) in MODES.items():
        selected = [(x, y) for m, x, y, _ in points if m == mode]
        if selected:
            ax.scatter(
                *zip(*selected, strict=True),
                s=50,
                color=color,
                marker=marker,
                edgecolors=SURFACE,
                linewidths=1,
                label=mode,
                zorder=3,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)", color=INK)
    ax.set_ylabel("Achieved TFLOPS", color=INK)
    ax.set_title(
        f"{profile.device.name}: {args.variant}, DRAM read {roofline.dram_read_gbps:.0f} GB/s",
        color=INK,
        fontsize=11,
        loc="left",
    )
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    if points:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, facecolor=SURFACE)
    print(f"wrote {args.output}: {len(points)} points")
    for problem, mode, gbps, fits in sorted(bandwidth):
        share = gbps / roofline.dram_read_gbps
        note = "  (bytes fit in L2; writes can stay there)" if fits else ""
        print(f"{problem:<40}{mode:<9}{gbps:8.0f} GB/s  {share:6.1%} of DRAM read{note}")


if __name__ == "__main__":
    main()
