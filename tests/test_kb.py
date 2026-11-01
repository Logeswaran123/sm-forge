from pathlib import Path

import pytest
import yaml

from smforge.kb.loader import KnowledgeBaseError, load_kb, record_measurement
from smforge.kb.schema import Measurement
from smforge.models import DeviceInfo, DeviceProfile, ProbeResult, Stage

CONSTRAINT = {
    "id": "dot_min_dim",
    "severity": "critical",
    "stages": ["autotuning"],
    "rule": "tl.dot operands need every dim >= 16.",
    "wrong": "BLOCK_K = 8",
    "correct": "BLOCK_K = 32",
    "check": {"problem": "p", "expect": "static", "match": "dot_min_dim"},
}
PATTERN = {
    "id": "group_m_swizzle",
    "stage": "device_specific",
    "applies_to": ["gemm", "matmul"],
    "summary": "Launch tiles in groups of GROUP_M rows.",
    "how": "Map a 1D program id to (pid_m, pid_n) in groups.",
    "example": "swizzle",
}
EXAMPLE = {"id": "swizzle", "problem": "p", "description": "2D grid to grouped 1D grid"}


def write_kb(root: Path, constraints=(CONSTRAINT,), patterns=(PATTERN,), examples=(EXAMPLE,)):
    (root / "patterns").mkdir(parents=True)
    (root / "constraints.yaml").write_text(yaml.safe_dump({"constraints": list(constraints)}))
    by_stage: dict[str, list] = {}
    for pattern in patterns:
        by_stage.setdefault(pattern["stage"], []).append(pattern)
    for stage, entries in by_stage.items():
        (root / "patterns" / f"{stage}.yaml").write_text(yaml.safe_dump({"patterns": entries}))
    (root / "examples").mkdir()
    (root / "examples" / "index.yaml").write_text(yaml.safe_dump({"examples": list(examples)}))
    for constraint in constraints:
        path = root / "checks" / constraint["id"]
        path.mkdir(parents=True)
        (path / "wrong.py").write_text("wrong\n")
        (path / "correct.py").write_text("correct\n")
    for example in examples:
        path = root / "examples" / example["id"]
        path.mkdir(parents=True)
        (path / "before.py").write_text("grid = (m, n)\nx = 1\n")
        (path / "after.py").write_text("grid = (m * n,)\nx = 1\n")
    return root


def profile(capability=(12, 0), probes=()) -> DeviceProfile:
    info = DeviceInfo(
        index=0,
        name="NVIDIA GeForce RTX 5080",
        capability=capability,
        sm_count=84,
        memory_bytes=16 << 30,
        l2_bytes=64 << 20,
        smem_per_block=101376,
        smem_per_sm=102400,
        regs_per_sm=65536,
        max_threads_per_sm=1536,
        warp_size=32,
        memory_bus_bits=256,
        torch="2.11.0",
        triton="3.6.0",
    )
    results = [ProbeResult(name=name, compiled=ok, lowered=ok) for name, ok in probes]
    return DeviceProfile(device=info, probes=results)


def test_load(tmp_path):
    kb = load_kb(write_kb(tmp_path))
    assert [c.id for c in kb.constraints] == ["dot_min_dim"]
    assert [p.id for p in kb.patterns] == ["group_m_swizzle"]
    assert kb.patterns[0].stage is Stage.DEVICE_SPECIFIC
    assert kb.examples["swizzle"].variant == "bench-gpu"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"patterns": (PATTERN, PATTERN)}, "group_m_swizzle: duplicate id"),
        ({"patterns": ({**PATTERN, "example": "nope"},)}, "unknown example nope"),
        ({"constraints": ({**CONSTRAINT, "id": "group_m_swizzle"},)}, "duplicate id"),
    ],
)
def test_invalid(tmp_path, change, message):
    root = write_kb(tmp_path, **change)
    with pytest.raises(KnowledgeBaseError, match=message):
        load_kb(root)


def test_missing_files_and_misfiled_stage(tmp_path):
    root = write_kb(tmp_path)
    (root / "examples" / "swizzle" / "after.py").unlink()
    (root / "patterns" / "device_specific.yaml").rename(root / "patterns" / "fusion.yaml")
    with pytest.raises(KnowledgeBaseError) as error:
        load_kb(root)
    assert "swizzle: missing" in str(error.value)
    assert "stage device_specific in fusion.yaml" in str(error.value)


def test_for_device(tmp_path):
    tma = {**PATTERN, "id": "tma", "requires": ["tma_device"]}
    hopper = {**PATTERN, "id": "wgmma", "arch": {"min": 90, "max": 90}}
    kb = load_kb(write_kb(tmp_path, patterns=(PATTERN, tma, hopper)))
    ids = [p.id for p in kb.for_device(profile(probes=[("tma_device", True)])).patterns]
    assert ids == ["group_m_swizzle", "tma"]
    assert [p.id for p in kb.for_device(profile(probes=[("tma_device", False)])).patterns] == [
        "group_m_swizzle"
    ]
    assert [p.id for p in kb.for_device(profile(capability=(9, 0))).patterns] == [
        "group_m_swizzle",
        "wgmma",
    ]
    assert kb.for_device(profile(capability=(7, 5))).patterns == []


def test_render(tmp_path):
    warning = {**CONSTRAINT, "id": "late", "severity": "warning", "stages": []}
    norm = {**PATTERN, "id": "row_norm", "applies_to": ["norm"]}
    generic = {**PATTERN, "id": "generic", "applies_to": ["any"]}
    kb = load_kb(write_kb(tmp_path, (warning, CONSTRAINT), (generic, PATTERN, norm)))
    text = kb.render(Stage.DEVICE_SPECIFIC, ["gemm"])
    assert text.index("**late**") < text.index("### generic")
    assert "**dot_min_dim**" not in text
    assert text.index("### group_m_swizzle") < text.index("### generic")
    assert "row_norm" not in text
    assert "-grid = (m, n)\n+grid = (m * n,)" in text
    assert "Example on p:" in text
    assert "row_norm" in kb.render(Stage.DEVICE_SPECIFIC)


def test_snippet_fence(tmp_path):
    triton = load_kb(write_kb(tmp_path / "triton")).render_all()
    cuda = load_kb(write_kb(tmp_path / "cuda")).render_all()
    assert "  ```python\n" in triton and "```cpp" not in triton
    assert "  ```cpp\n" in cuda and "```python" not in cuda


def test_render_budget(tmp_path):
    kb = load_kb(write_kb(tmp_path))
    full = kb.render(Stage.DEVICE_SPECIFIC)
    assert "```diff" in full
    tight = kb.render(Stage.DEVICE_SPECIFIC, budget=(len(full) - 40) // 4)
    assert "### group_m_swizzle" in tight
    assert "```diff" not in tight
    assert kb.render(Stage.DEVICE_SPECIFIC, budget=10) == ""


def test_record_measurement(tmp_path):
    kb = load_kb(write_kb(tmp_path))
    first = Measurement(
        gpu="rtx5080",
        variant="bench-gpu",
        before_us=2.0,
        after_us=1.0,
        speedup=2.0,
        triton="3.6.0",
        date="2026-10-27",
    )
    record_measurement(kb, "swizzle", first)
    record_measurement(kb, "swizzle", first.model_copy(update={"speedup": 1.5}))
    record_measurement(kb, "swizzle", first.model_copy(update={"gpu": "h100"}))
    reloaded = load_kb(tmp_path).examples["swizzle"].measured
    assert [(m.gpu, m.speedup) for m in reloaded] == [("rtx5080", 1.5), ("h100", 2.0)]


def test_unused_example(tmp_path):
    root = write_kb(tmp_path, examples=(EXAMPLE, {**EXAMPLE, "id": "orphan"}))
    with pytest.raises(KnowledgeBaseError, match="orphan: no pattern uses this example"):
        load_kb(root)
