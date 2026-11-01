import pytest
from test_optimize import slow
from test_seed import PROBLEM, copy_problem

from smforge.config import Config, TimingConfig
from smforge.harness.runner import Runner
from smforge.models import SanitizerReport
from smforge.trials import TrialError, TrialTree

pytestmark = pytest.mark.gpu


def test_trial_tree(tmp_path):
    problem = copy_problem(tmp_path)
    good = (PROBLEM / "kernel.py").read_text()
    problem.seed.write_text(slow(good))
    work = tmp_path / "work"
    work.mkdir()
    config = Config(cache_dir=tmp_path / "cache", timing=TimingConfig(warmup=3, iters=40, rounds=4))
    with Runner(config) as runner:
        ledger = tmp_path / "lessons.md"
        tree = TrialTree.create(tmp_path / "trials", problem, problem.seed, runner, ledger)
        with pytest.raises(TrialError):
            TrialTree.create(tmp_path / "trials", problem, problem.seed, runner)
        (work / "same.py").write_text(slow(good))
        same = tree.save(work / "same.py", "t0", "copy of the seed")
        assert tree.evaluate(same, runner).kind in ("parity", "slower")
        (work / "wrong.py").write_text(good.replace("acc + bias", "acc - bias"))
        wrong = tree.save(work / "wrong.py", "t1", "subtract the bias")
        assert tree.evaluate(wrong, runner).kind == "mismatch"
        assert tree.finalize(tmp_path / "out.py") is None
        (work / "fast.py").write_text(good)
        fast = tree.save(work / "fast.py", "t1", "64x64 tiles")
        outcome = tree.evaluate(fast, runner)
    assert outcome.kind == "accepted" and fast.speedup > 1.5
    tree = TrialTree(tmp_path / "trials")
    assert [t.id for t in tree.ranked()] == ["t3", "t1"]
    assert "t3: accepted" in tree.status() and "<- best" in tree.status()

    flagged = SanitizerReport(tool="memcheck", ok=False, errors=4)
    assert tree.finalize(tmp_path / "out.py", lambda path: flagged) is None
    assert tree.trials["t3"].memcheck == "4 errors"
    tree.trials["t3"].memcheck = None
    clean = SanitizerReport(tool="memcheck", ok=True, errors=0)
    assert tree.finalize(tmp_path / "out.py", lambda path: clean).id == "t3"
    assert (tmp_path / "out.py").read_text() == good
    entries = [line for line in ledger.read_text().splitlines() if line.startswith("- ")]
    assert entries[1] == "- t2 (from t1): mismatch: subtract the bias"
    assert entries[2].startswith("- t3 (from t1): accepted ") and entries[2].endswith(
        ": 64x64 tiles"
    )
    assert entries[-1].startswith("- finalized t3")
    tree.path(tree.trials["t3"]).write_text(good + "\n# edited\n")
    assert tree.best().id == "t1"
    with pytest.raises(TrialError, match="no trial t9"):
        tree.save(work / "fast.py", "t9")
