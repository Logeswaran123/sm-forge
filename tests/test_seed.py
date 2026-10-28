import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from anthropic.types.beta import BetaTextBlock

from smforge.agents.cover import NO_CODE, cover, extract_code
from smforge.agents.seed import seed_problem, system_prompt, task_prompt
from smforge.config import Config, LLMConfig, TimingConfig
from smforge.corpus.catalog import Problem
from smforge.harness.runner import Runner
from smforge.llm.client import BudgetExceeded, LLMClient, Refused
from smforge.spec import load_spec

PROBLEM = Path(__file__).parent / "kernels" / "linear_relu"


def message(text: str, stop_reason: str = "end_turn", tokens: tuple = (1000, 500, 0, 0)):
    return SimpleNamespace(
        content=[BetaTextBlock(type="text", text=text)],
        model="claude-opus-5-5",
        stop_reason=stop_reason,
        stop_details=None,
        usage=SimpleNamespace(
            input_tokens=tokens[0],
            output_tokens=tokens[1],
            cache_creation_input_tokens=tokens[2],
            cache_read_input_tokens=tokens[3],
        ),
    )


def scripted(monkeypatch, llm: LLMClient, replies: list[str]) -> list[dict]:
    requests = []

    def send(request):
        requests.append({**request, "messages": list(request["messages"])})
        return message(replies[len(requests) - 1])

    monkeypatch.setattr(llm, "_send", send)
    return requests


def test_extract_code():
    assert extract_code("a\n```python\nx = 1\n```\nb\n```py\ny = 2\n```") == "y = 2\n"
    assert extract_code("no code") is None


def test_cover_feeds_failures_back(monkeypatch, tmp_path):
    llm = LLMClient(LLMConfig(), transcripts=tmp_path)
    requests = scripted(
        monkeypatch, llm, ["thinking aloud", "```python\nbad\n```", "```python\ngood\n```"]
    )
    result = cover(llm, "system", "task", lambda code, _: (code == "good\n", "wrong"), max_iters=4)
    assert result.passed and result.code == "good\n"
    assert [a.feedback for a in result.attempts] == [NO_CODE, "wrong", "wrong"]
    assert [a.passed for a in result.attempts] == [False, False, True]
    assert [len(r["messages"]) for r in requests] == [1, 3, 5]
    assert "wrong" in requests[2]["messages"][-1]["content"]
    assert requests[0]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert requests[0]["fallbacks"] == "default"
    assert "temperature" not in requests[0]
    assert result.usage.calls == 3
    assert result.usage.cost_usd == pytest.approx(3 * (1000 * 4 + 500 * 20) / 1e6)
    assert len(list(tmp_path.glob("*.json"))) == 3


def test_cover_keeps_recent_attempts(monkeypatch):
    llm = LLMClient(LLMConfig())
    requests = scripted(monkeypatch, llm, [f"```python\nv{i}\n```" for i in range(4)])
    result = cover(llm, "system", "task", lambda code, _: (False, code), max_iters=4, keep=1)
    assert not result.passed
    assert [len(r["messages"]) for r in requests] == [1, 3, 3, 3]
    assert requests[3]["messages"][0]["content"] == "task"
    assert "v2" in requests[3]["messages"][-1]["content"]


def test_budget_and_refusal(monkeypatch):
    llm = LLMClient(LLMConfig(budget_usd=0.0))
    with pytest.raises(BudgetExceeded):
        llm.complete("system", [{"role": "user", "content": "x"}])
    llm = LLMClient(LLMConfig())
    monkeypatch.setattr(llm, "_send", lambda request: message("", stop_reason="refusal"))
    with pytest.raises(Refused):
        llm.complete("system", [{"role": "user", "content": "x"}])


def copy_problem(tmp_path: Path) -> Problem:
    path = tmp_path / "linear_relu"
    path.mkdir()
    for name in ("reference.py", "spec.yaml"):
        shutil.copyfile(PROBLEM / name, path / name)
    return Problem(id="linear_relu", source="test", family="gemm", path=path)


def test_prompts(tmp_path):
    problem = copy_problem(tmp_path)
    task = task_prompt(problem, load_spec(problem.spec))
    assert "self.linear = nn.Linear" in task
    assert "- `ci`: Model(96, 200), inputs X [127, 96] float16; rtol 0.01, atol 0.01" in task
    assert "$device" not in system_prompt(None)


@pytest.mark.gpu
def test_seed_problem(monkeypatch, tmp_path):
    problem = copy_problem(tmp_path)
    good = (PROBLEM / "kernel.py").read_text()
    fallback = good.replace(
        "        return y\n", "        return torch.relu(x @ self.weight.t() + self.bias)\n"
    )
    config = Config(cache_dir=tmp_path / "cache", timing=TimingConfig(warmup=2, iters=4, rounds=2))
    llm = LLMClient(LLMConfig())
    scripted(monkeypatch, llm, [f"```python\n{fallback}```", f"```python\n{good}```"])
    with Runner(config) as runner:
        result = seed_problem(problem, runner, llm, None, 3, tmp_path / "work")
    assert result.passed
    assert "Static checks" in result.attempts[0].feedback
    assert problem.seed.read_text() == good
    assert yaml.safe_load((problem.path / "seed.yaml").read_text())["attempts"] == 2
    assert len(json.loads((tmp_path / "work" / "attempts.json").read_text())) == 2
