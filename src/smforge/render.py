"""Plain-text rendering of results for the CLIs."""

from __future__ import annotations

from smforge.harness.audit import short_name
from smforge.models import (
    CheckResult,
    Comparison,
    DeviceProfile,
    FactsReport,
    LaunchProfile,
    Severity,
    Timing,
    ValidationIssue,
)


def issues(items: list[ValidationIssue]) -> str:
    if not items:
        return "no issues"
    lines = []
    for issue in items:
        where = f"line {issue.line}: " if issue.line else ""
        lines.append(f"{issue.severity:<8}{issue.check:<22}{where}{issue.message}")
    errors = sum(issue.severity is Severity.ERROR for issue in items)
    lines.append(f"errors: {errors}, warnings: {len(items) - errors}")
    return "\n".join(lines)


def check(result: CheckResult) -> str:
    tolerance = f"rtol {result.rtol:g}, atol {result.atol:g}"
    if result.passed:
        return f"correctness: pass ({result.draws} draws, {tolerance}, weights {result.weights})"
    lines = [f"correctness: FAIL at draw {result.draws} ({tolerance})"]
    if result.error:
        lines.append(f"  {result.error.strip().splitlines()[0]}")
    for diff in result.diffs:
        if not diff.mismatched:
            continue
        line = (
            f"  output {diff.index} {diff.shape}: {diff.mismatched}/{diff.total} mismatched, "
            f"max abs {diff.max_abs:.3g}, max rel {diff.max_rel:.3g}, first at {diff.first_bad}"
        )
        if diff.nan or diff.inf:
            line += f", {diff.nan} NaN, {diff.inf} Inf"
        lines.append(line)
    return "\n".join(lines)


def launches(profile: LaunchProfile) -> str:
    lines = [f"launches per forward, {profile.kernel_us:.1f} us of kernel time:"]
    for launch in profile.launches:
        lines.append(
            f"  {launch.category:<13}{launch.calls:>4g}x {launch.duration_us:>10.1f} us  "
            f"{short_name(launch.name, 80)}"
        )
    return "\n".join(lines)


def timing(result: Timing) -> str:
    mode = ", CUDA graph" if result.graph else ""
    lines = [
        f"median {result.median_us:.1f} us, mean {result.mean_us:.1f} us, "
        f"min {result.min_us:.1f} us, max {result.max_us:.1f} us ({result.samples} samples{mode})"
    ]
    rates = []
    if result.tflops:
        rates.append(f"{result.tflops:.1f} TFLOPS")
    if result.gbps:
        rates.append(f"{result.gbps:.0f} GB/s")
    if rates:
        lines.append(", ".join(rates))
    return "\n".join(lines)


def comparison(result: Comparison) -> str:
    return (
        f"baseline {result.baseline_us:.1f} us, candidate {result.candidate_us:.1f} us: "
        f"{result.speedup:.3f}x, {result.verdict} (min effect {result.min_effect:.1%})\n"
        f"rounds: {' '.join(f'{r:.3f}' for r in result.ratios)}"
    )


def facts(report: FactsReport, ridge: float | None = None) -> str:
    if not report.kernels:
        return "no Triton kernel launched"
    lines = []
    for kernel in report.kernels:
        grid = "x".join(map(str, kernel.grid))
        lines.append(
            f"{kernel.name}: grid {grid} ({kernel.ctas} CTAs), {kernel.num_warps} warps, "
            f"{kernel.num_stages} stages"
        )
        occupancy = ""
        if kernel.ctas_per_sm:
            occupancy = f", {kernel.ctas_per_sm} CTAs/SM, {kernel.waves:.2f} waves"
        lines.append(
            f"  regs {kernel.n_regs}, spills {kernel.n_spills}, "
            f"smem {kernel.shared_bytes / 1024:.1f} KB{occupancy}"
        )
        counts = ", ".join(f"{op} {n}" for op, n in {**kernel.mma, **kernel.instructions}.items())
        lines.append(f"  sass: {kernel.total_instructions} instructions; {counts}")
    if report.intensity:
        line = f"arithmetic intensity {report.intensity:.1f} FLOP/byte"
        if ridge:
            side = "compute" if report.intensity >= ridge else "memory"
            line += f"; ridge point {ridge:.1f}, so {side}-bound"
        lines.append(line)
    return "\n".join(lines)


def device(profile: DeviceProfile) -> str:
    d = profile.device
    wsl = ", WSL2" if d.wsl else ""
    lines = [
        f"{d.name} ({d.arch}), {d.sm_count} SMs, {d.memory_bytes / 2**30:.1f} GiB, "
        f"L2 {d.l2_bytes >> 20} MB",
        f"  shared memory {d.smem_per_block // 1024} KB/block, {d.smem_per_sm // 1024} KB/SM; "
        f"{d.regs_per_sm} registers/SM; {d.max_threads_per_sm} threads/SM; "
        f"{d.max_ctas_per_sm} CTAs/SM",
        f"  max clocks: SM {d.max_sm_clock_mhz} MHz, memory {d.max_memory_clock_mhz} MHz; "
        f"power limit {d.power_limit_w} W",
        f"  driver {d.driver}, CUDA {d.cuda}, torch {d.torch}, triton {d.triton}{wsl}",
    ]
    if profile.probes:
        lines.append("probes:")
        for probe in profile.probes:
            detail = ", ".join(f"{op} {n}" for op, n in probe.instructions.items())
            if not probe.compiled:
                detail = probe.error or "did not compile"
            elif probe.correct is False:
                detail += " (wrong result)"
            lines.append(f"  {probe.name:<17}{'yes' if probe.supported else 'no':<5}{detail}")
    if profile.roofline:
        r = profile.roofline
        lines.append(
            f"roofline: DRAM copy {r.dram_copy_gbps:.0f} GB/s, DRAM read {r.dram_read_gbps:.0f} "
            f"GB/s, L2 copy {r.l2_copy_gbps:.0f} GB/s"
        )
        for dtype, tflops in r.tflops.items():
            lines.append(
                f"  {dtype:<15}{tflops:>7.1f} TFLOPS, ridge {r.ridge(dtype):.0f} FLOP/byte"
            )
    if profile.noise:
        lines.append("A/A noise:")
        for case in profile.noise.cases:
            lines.append(
                f"  {case.name:<15}{case.median_us:>9.1f} us, deviation {case.deviation:.2%}, "
                f"round spread {case.spread:.2%}"
            )
        lines.append(f"  suggested BENCHMARK_MIN_EFFECT={profile.noise.min_effect}")
    if profile.tools:
        lines.append("tools:")
        for tool in profile.tools:
            status = "ok" if tool.ok else "unavailable"
            lines.append(f"  {tool.name:<19}{status:<13}{tool.detail}")
    return "\n".join(lines)
