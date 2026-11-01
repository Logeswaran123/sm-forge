import json
from pathlib import Path

import pytest

from smforge.config import get_config
from smforge.skill_cli import main

PROBLEM = Path(__file__).parent / "kernels" / "linear_relu"
SPEC = str(PROBLEM / "spec.yaml")
KERNEL = str(PROBLEM / "kernel.py")
REFERENCE = str(PROBLEM / "reference.py")


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HARDWARE_DIR", str(tmp_path / "hardware"))
    monkeypatch.setenv("BENCH_WARMUP", "5")
    monkeypatch.setenv("BENCH_ITERS", "20")
    monkeypatch.setenv("BENCH_ROUNDS", "2")
    get_config.cache_clear()
    yield
    get_config.cache_clear()


def test_validate(capsys):
    assert main(["validate", KERNEL, "--spec", SPEC]) == 0
    assert capsys.readouterr().out.strip() == "no issues"
    assert main(["validate", REFERENCE, "--spec", SPEC, "--json"]) == 1
    checks = {issue["check"] for issue in json.loads(capsys.readouterr().out)}
    assert checks == {
        "missing_triton_import",
        "missing_triton_kernel",
        "torch_compute_call",
        "torch_module_call",
    }


def test_check_needs_an_oracle(capsys):
    assert main(["check", KERNEL]) == 2


@pytest.mark.gpu
def test_check_and_bench(capsys):
    assert main(["check", KERNEL, "--reference", REFERENCE, "--spec", SPEC, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["check"]["passed"]
    assert report["audit"] == []
    assert main(["check", REFERENCE, "--baseline", KERNEL, "--spec", SPEC, "--variant", "ci"]) == 1
    assert "no_own_launch" in capsys.readouterr().out
    assert main(["bench", KERNEL, "--baseline", KERNEL, "--spec", SPEC, "--variant", "ci"]) == 0
    assert "x," in capsys.readouterr().out
