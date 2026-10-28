"""Every knowledge base entry's kernel pair on this GPU.

Run from the repository root: ``pytest runners/test_kb_examples.py``. ``smforge kb verify --write``
runs the same checks and records the example speedups.
"""

import shutil

import pytest

from smforge.corpus.catalog import load_problems, load_splits
from smforge.harness.runner import Runner
from smforge.kb.loader import load_kb
from smforge.kb.verify import verify_constraint, verify_example
from smforge.models import DeviceInfo

pytestmark = pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="no NVIDIA GPU")

KB = load_kb()
PROBLEMS = {p.id: p for p in load_problems()}
DEV = set(load_splits().get("dev", []))


@pytest.fixture(scope="module")
def runner():
    with Runner() as runner:
        yield runner


@pytest.mark.parametrize("constraint", KB.constraints, ids=lambda c: c.id)
def test_constraint(runner, constraint):
    outcome = verify_constraint(KB, constraint, runner, PROBLEMS, DEV)
    assert outcome.passed, outcome.detail


@pytest.mark.parametrize("example", list(KB.examples.values()), ids=lambda e: e.id)
def test_example(runner, example):
    device = DeviceInfo.model_validate(runner.call("device"))
    outcome = verify_example(KB, example, runner, PROBLEMS, DEV, device)
    assert outcome.passed, outcome.detail
