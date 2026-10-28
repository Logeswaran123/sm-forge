import logging

import pytest
from anthropic.types.beta import BetaTextBlock
from test_seed import message, scripted

from smforge.config import LLMConfig
from smforge.llm.client import (
    LLMClient,
    Recorder,
    ReplayClient,
    ReplayMiss,
    create_client,
    prompt_key,
)


def test_prompt_key_ignores_block_type():
    sdk = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": [BetaTextBlock(type="text", text="reply")]},
    ]
    plain = [sdk[0], {"role": "assistant", "content": [{"type": "text", "text": "reply"}]}]
    assert prompt_key("system", sdk) == prompt_key("system", plain)
    assert prompt_key("system", sdk) != prompt_key("system", sdk, role="planner")
    assert prompt_key("system", sdk) != prompt_key("other", sdk)


def test_effort_per_role(monkeypatch):
    llm = LLMClient(LLMConfig(efforts={"planner": "low"}))
    requests = scripted(monkeypatch, llm, ["a", "b"])
    llm.complete("system", [{"role": "user", "content": "x"}], role="planner")
    llm.complete("system", [{"role": "user", "content": "x"}], role="optimizer")
    assert [r["output_config"]["effort"] for r in requests] == ["low", "high"]


def test_record_and_replay(monkeypatch, tmp_path):
    path = tmp_path / "replies.jsonl"
    llm = LLMClient(LLMConfig())
    scripted(monkeypatch, llm, ["first", "second"])
    recorder = Recorder(llm, path)
    task = [{"role": "user", "content": "task"}]
    first = recorder.complete("system", task)
    turn = [
        *task,
        {"role": "assistant", "content": first.content},
        {"role": "user", "content": "more"},
    ]
    recorder.complete("system", turn)

    replay = create_client(LLMConfig(), replay=path)
    assert isinstance(replay, ReplayClient)
    again = replay.complete("system", task)
    assert again.text == "first"
    turn = [
        *task,
        {"role": "assistant", "content": again.content},
        {"role": "user", "content": "more"},
    ]
    assert replay.complete("system", turn).text == "second"
    assert replay.usage.calls == 2
    with pytest.raises(ReplayMiss):
        replay.complete("system", [{"role": "user", "content": "unseen"}])


def test_cache_miss_is_logged(monkeypatch, caplog):
    llm = LLMClient(LLMConfig())
    monkeypatch.setattr(llm, "_send", lambda request: message("ok", tokens=(10, 5, 0, 0)))
    turn = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    with caplog.at_level(logging.WARNING):
        llm.complete("system", turn[:1])
        assert not caplog.records
        llm.complete("system", turn)
    assert "read nothing from the prompt cache" in caplog.text
