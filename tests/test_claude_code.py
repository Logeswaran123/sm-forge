import json
from types import SimpleNamespace

import pytest

from smforge.agents.cover import cover
from smforge.config import LLMConfig
from smforge.llm import claude_code
from smforge.llm.claude_code import ClaudeCodeClient, ClaudeCodeError
from smforge.llm.client import LLMClient, create_client


def result(text: str, session: str = "s1", cost: float = 0.25, error: bool = False) -> str:
    return json.dumps(
        {
            "is_error": error,
            "result": text,
            "session_id": session,
            "stop_reason": "end_turn",
            "total_cost_usd": cost,
            "usage": {"input_tokens": 10, "output_tokens": 20, "cache_read_input_tokens": 5},
        }
    )


@pytest.fixture
def calls(monkeypatch):
    log = []
    replies = []

    def run(command, input, **kwargs):
        log.append({"command": command, "input": input, "cwd": kwargs["cwd"]})
        return SimpleNamespace(stdout=replies.pop(0), stderr="", returncode=0)

    monkeypatch.setattr(claude_code.subprocess, "run", run)
    return log, replies


def test_backend_selection():
    assert isinstance(create_client(LLMConfig()), LLMClient)
    assert isinstance(create_client(LLMConfig(backend="claude-code")), ClaudeCodeClient)
    with pytest.raises(ValueError):
        create_client(LLMConfig(backend="nope"))


def test_session_resumes_with_only_the_new_turn(calls, tmp_path):
    log, replies = calls
    replies += [result("first"), result("```python\ngood\n```")]
    llm = ClaudeCodeClient(LLMConfig(backend="claude-code", claude_bin="claude"), tmp_path)
    outcome = cover(llm, "system", "task", lambda code, _: (True, "ok"), max_iters=3)
    assert outcome.passed
    first, second = (entry["command"] for entry in log)
    assert first[first.index("--tools") + 1] == ""
    assert first[first.index("--model") + 1] == "claude-opus-5-5"
    assert first[first.index("--system-prompt") + 1] == "system"
    assert "--resume" not in first
    assert second[second.index("--resume") + 1] == "s1"
    assert log[0]["input"] == "task"
    assert log[1]["input"].startswith("Verification failed")
    assert not any(log[0]["cwd"].iterdir())
    assert llm.usage.calls == 2
    assert llm.usage.cost_usd == pytest.approx(0.5)
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_new_conversation_starts_a_new_session(calls):
    log, replies = calls
    replies += [result("a", session="s1"), result("b", session="s2")]
    llm = ClaudeCodeClient(LLMConfig(backend="claude-code", claude_bin="claude"))
    llm.complete("system", [{"role": "user", "content": "one"}])
    llm.complete("system", [{"role": "user", "content": "two"}])
    assert "--resume" not in log[1]["command"]


def test_errors(calls):
    _, replies = calls
    replies += [result("API Error: 400 needs a newer version", error=True), "not json"]
    llm = ClaudeCodeClient(LLMConfig(backend="claude-code", claude_bin="claude"))
    with pytest.raises(ClaudeCodeError, match="newer version"):
        llm.complete("system", [{"role": "user", "content": "x"}])
    with pytest.raises(ClaudeCodeError, match="exited with code 0"):
        llm.complete("system", [{"role": "user", "content": "x"}])
