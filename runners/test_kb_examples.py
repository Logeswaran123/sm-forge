"""Every knowledge base entry's kernel pair on this GPU, Triton and CUDA.

Run from the repository root: ``pytest runners/test_kb_examples.py``. ``smforge kb verify --write``
runs the same checks and records the example speedups.
"""

import shutil

import pytest

from smforge.corpus.catalog import load_problems, load_splits
from smforge.harness.runner import Runner
from smforge.kb.loader import kb_dir, load_kb
from smforge.kb.verify import verify_constraint, verify_example
from smforge.models import DeviceInfo, Dsl

pytestmark = pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="no NVIDIA GPU")

KBS = {dsl: load_kb(kb_dir(dsl)) for dsl in Dsl}
CONSTRAINTS = [(kb, c) for kb in KBS.values() for c in kb.constraints]
EXAMPLES = [(kb, e) for kb in KBS.values() for e in kb.examples.values()]
PROBLEMS = {p.id: p for p in load_problems()}
DEV = set(load_splits().get("dev", []))


@pytest.fixture(scope="module")
def runner():
    with Runner() as runner:
        yield runner


@pytest.mark.parametrize(
    ("kb", "constraint"), CONSTRAINTS, ids=[f"{kb.root.name}/{c.id}" for kb, c in CONSTRAINTS]
)
def test_constraint(runner, kb, constraint):
    outcome = verify_constraint(kb, constraint, runner, PROBLEMS, DEV)
    assert outcome.passed, outcome.detail


@pytest.mark.parametrize(
    ("kb", "example"), EXAMPLES, ids=[f"{kb.root.name}/{e.id}" for kb, e in EXAMPLES]
)
def test_example(runner, kb, example):
    device = DeviceInfo.model_validate(runner.call("device"))
    outcome = verify_example(kb, example, runner, PROBLEMS, DEV, device)
    assert outcome.passed, outcome.detail
