import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_optimize import slow
from test_seed import PROBLEM, copy_problem

import smforge.claude_ws as ws
from smforge.claude_ws import PERMISSIONS, Session, parse_session, run_claude, settings
from smforge.config import Config, TimingConfig
from smforge.harness.runner import Runner
from smforge.kb.loader import load_kb


def test_settings():
    assert settings(Path("/w"))["autoMemoryEnabled"] is False
    rules = settings(Path("/w"))["permissions"]
    assert "Edit(//w/work/**)" in rules["allow"]
    assert "Bash(smforge-skill trial save *)" in rules["allow"]
    assert "Edit(//w/trials/**)" in rules["deny"] and "Edit(//w/kernel.py)" in rules["deny"]
    assert "Bash(smforge-skill trial finalize *)" in rules["deny"]
    assert not any(rule.startswith("Bash(python") for rule in rules["allow"])


def test_parse_session(tmp_path):
    path = tmp_path / "transcript.jsonl"
    events = [
        {"type": "system", "subtype": "init"},
        {"type": "result", "subtype": "error_max_turns", "num_turns": 9, "total_cost_usd": 1.5},
    ]
    path.write_text("Ignoring 3 entries\n" + "\n".join(json.dumps(e) for e in events) + "\n")
    session = parse_session(path)
    assert (session.stop, session.turns, session.cost_usd) == ("error_max_turns", 9, 1.5)
    path.write_text("claude: not logged in\n")
    with pytest.raises(ws.ClaudeSessionError, match="not logged in"):
        parse_session(path)


@pytest.mark.gpu
def test_run_claude(monkeypatch, tmp_path):
    problem = copy_problem(tmp_path)
    good = (PROBLEM / "kernel.py").read_text()
    problem.seed.write_text(slow(good))

    def session(root, config, max_turns, budget_usd):
        """What the agent does: write a kernel under work/ and save it as a trial."""
        assert (root / PERMISSIONS).exists() and (
            root / ".claude/commands/optimize-kernel.md"
        ).exists()
        assert "## Measurements of `kernel.py` (trial `t0`)" in (root / "CLAUDE.md").read_text()
        assert not os.access(root / "kernel.py", os.W_OK)
        (root / "work" / "fast.py").write_text(good)
        skill = Path(sys.executable).parent / "smforge-skill"
        command = [str(skill), "trial", "save", "work/fast.py", "--strategy", "64x64 tiles"]
        env = {**os.environ, "CACHE_DIR": str(config.cache_dir), "BENCH_ITERS": "40"}
        out = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
        assert out.returncode == 0, out.stdout + out.stderr
        assert out.stdout.startswith("t1 (from t0): accepted")
        return Session("success", 7, 0.5, "done", False)

    monkeypatch.setattr(ws, "run_session", session)
    config = Config(cache_dir=tmp_path / "cache", timing=TimingConfig(warmup=3, iters=40, rounds=4))
    with Runner(config) as runner:
        result = run_claude(
            problem,
            runner,
            load_kb(),
            None,
            config,
            tmp_path / "ws",
            4,
            20,
            memcheck=False,
            ncu=False,
        )
    assert result.final == "t1" and result.trials == 1 and len(result.ab) == 3
    assert result.speedup == pytest.approx(min(result.ab), rel=1e-3)
    assert result.speedup > 1.5 and result.trial_speedup > 1.5
    assert (result.workspace / "output.py").read_text() == good


@pytest.mark.gpu
def test_session_error_stops_the_run(monkeypatch, tmp_path):
    problem = copy_problem(tmp_path)
    problem.seed.write_text((PROBLEM / "kernel.py").read_text())
    failed = Session("success", 1, 0.0, "API Error: 400 version 2.1.280 or newer is required", True)
    monkeypatch.setattr(ws, "run_session", lambda *args: failed)
    config = Config(cache_dir=tmp_path / "cache", timing=TimingConfig(warmup=3, iters=20, rounds=2))
    with (
        Runner(config) as runner,
        pytest.raises(ws.ClaudeSessionError, match="before its first trial"),
    ):
        run_claude(problem, runner, load_kb(), None, config, tmp_path / "ws", 4, 20, ncu=False)
