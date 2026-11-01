import csv
import io

from smforge.harness.ncu import diff_profiles, parse_details, render_profile

HEADER = [
    "ID",
    "Kernel Name",
    "Block Size",
    "Grid Size",
    "Section Name",
    "Metric Name",
    "Metric Unit",
    "Metric Value",
    "Rule Name",
    "Rule Type",
    "Rule Description",
    "Estimated Speedup Type",
    "Estimated Speedup",
]


def details(launch: int, kernel: str, duration: tuple[str, str], conflicts: bool) -> list[list]:
    def metric(section, name, unit, value):
        return [str(launch), kernel, "(128, 1, 1)", "(512, 1, 1)", section, name, unit, value]

    def rule(section, name, kind, text, scope="", speedup=""):
        return [*metric(section, "", "", ""), name, kind, text, scope, speedup]

    rows = [
        metric("GPU Speed Of Light Throughput", "Duration", duration[1], duration[0]),
        metric("GPU Speed Of Light Throughput", "DRAM Throughput", "%", "61.92"),
        metric("Memory Workload Analysis", "Memory Throughput", "Gbyte/s", "376.12"),
        metric("Launch Statistics", "Registers Per Thread", "register/thread", "40"),
        metric("Occupancy", "Achieved Occupancy", "%", "73.53"),
        rule("SpeedOfLight", "SOLBottleneck", "INF", "Memory is busy. More."),
        rule("SchedulerStats", "IssueSlotUtilization", "OPT", "Local.", "local", "90"),
    ]
    if conflicts:
        text = "Shared loads cause a 32-way bank conflict.This results in 6 conflicts."
        rows.append(
            rule(
                "MemoryWorkloadAnalysis_Tables",
                "SharedMemoryConflicts",
                "OPT",
                text,
                "global",
                "87.21",
            )
        )
    return rows


def table(*launches: list[list]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL)
    writer.writerow(HEADER)
    for rows in launches:
        writer.writerows(rows)
    return "==PROF== Disconnected\n" + out.getvalue()


def test_parse_and_render():
    text = table(details(0, "k", ("1.42", "ms"), True), details(1, "k", ("9.0", "ms"), False))
    (profile,) = parse_details(text)
    assert profile.kernel == "k" and profile.grid == "(512, 1, 1)"
    assert len(profile.metrics) == 5 and len(profile.rules) == 3
    text = render_profile(profile)
    assert "  speed of light: duration 1420.0 us, DRAM 61.92%" in text
    assert "  memory: throughput 376.1 GB/s" in text
    assert "  launch: registers 40" in text
    assert "SharedMemoryConflicts, up to 87%: Shared loads cause a 32-way bank conflict." in text
    assert "IssueSlotUtilization" not in text and "This results" not in text


def test_diff():
    before = parse_details(table(details(0, "k", ("1.42", "ms"), True)))
    after = parse_details(table(details(0, "k", ("997.63", "us"), False)))
    text = diff_profiles(before, after)
    assert "  speed of light:\n    duration: 1420.0 us -> 997.6 us (-30%)\n  ncu" in text
    assert "ncu estimates gone: SharedMemoryConflicts" in text
    assert "DRAM" not in text
    live = parse_details(table(details(0, "k", ("997630", "ns"), False)))
    assert diff_profiles(after, live) == "k: no change"
    renamed = parse_details(table(details(0, "k2", ("1.42", "ms"), True)))
    assert diff_profiles(before, renamed) == "k -> k2: no change"
