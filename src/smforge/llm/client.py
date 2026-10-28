"""The Anthropic client: model, effort, caching, refusals, usage, cost, transcripts and replay.

The system prompt carries an explicit cache breakpoint and the request enables automatic
caching, so a multi-turn exchange re-reads its history from the cache; a later turn that reads
nothing from the cache is logged. A classifier decline is retried server-side on the model
Anthropic recommends for that category (``fallbacks: "default"``); a decline that survives that
raises ``Refused``.

``Recorder`` appends every reply to a JSONL file keyed by a hash of the prompt, and
``ReplayClient`` answers from such a file without network access, so tests and reruns of a stage
need no API calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Protocol

import anthropic
from pydantic import BaseModel

from smforge.config import LLMConfig

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
_MEASURED = re.compile(r"\d+\.\d+(?:e[-+]?\d+)?|\d+(?= ?(?:GB/s|TFLOPS))")
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 5.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 2.50, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.01),
}
"""USD per million tokens: input, output, 5-minute cache write, cache read."""


class Usage(BaseModel):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: Usage) -> None:
        for field in Usage.model_fields:
            setattr(self, field, getattr(self, field) + getattr(other, field))


class Reply(BaseModel):
    text: str
    content: list[Any]
    model: str
    stop_reason: str | None
    usage: Usage


class Completer(Protocol):
    config: LLMConfig
    usage: Usage

    def complete(
        self, system: str, messages: list[dict], role: str | None = None, schema: dict | None = None
    ) -> Reply: ...


class Refused(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


class MissingCredentials(RuntimeError):
    pass


class ReplayMiss(KeyError):
    pass


class LLMClient:
    def __init__(self, config: LLMConfig, transcripts: Path | None = None):
        self.config = config
        self.transcripts = transcripts
        self.usage = Usage()
        self._client: anthropic.Anthropic | None = None

    def complete(
        self, system: str, messages: list[dict], role: str | None = None, schema: dict | None = None
    ) -> Reply:
        check_budget(self.config, self.usage)
        output_config: dict[str, Any] = {"effort": self.config.effort_for(role)}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        request = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
            "cache_control": {"type": "ephemeral"},
            "output_config": output_config,
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }
        message = self._send(request)
        reply = Reply(
            text="".join(block.text for block in message.content if block.type == "text"),
            content=list(message.content),
            model=message.model,
            stop_reason=message.stop_reason,
            usage=_usage(message.usage, message.model, self.config.model),
        )
        self.usage.add(reply.usage)
        record_transcript(self.transcripts, request, reply)
        if len(messages) > 1 and reply.usage.cache_read_tokens == 0:
            log.warning("turn %d read nothing from the prompt cache", len(messages) // 2 + 1)
        if reply.stop_reason == "refusal":
            raise Refused(f"{message.model} declined: {message.stop_details}")
        return reply

    def _send(self, request: dict) -> Any:
        if self._client is None:
            self._client = anthropic.Anthropic(max_retries=self.config.max_retries)
        try:
            with self._client.beta.messages.stream(**request) as stream:
                return stream.get_final_message()
        except anthropic.AuthenticationError as e:
            raise MissingCredentials(f"Anthropic rejected the credentials: {e}") from e
        except TypeError as e:
            if "authentication" not in str(e):
                raise
            raise MissingCredentials(
                "no Anthropic credentials: set ANTHROPIC_API_KEY in .env or run `ant auth login`"
            ) from e


class Recorder:
    """Passes calls to ``client`` and appends each reply to ``path`` under its prompt key."""

    def __init__(self, client: Completer, path: Path):
        self.client = client
        self.path = path
        self.config = client.config
        self.usage = client.usage

    def complete(
        self, system: str, messages: list[dict], role: str | None = None, schema: dict | None = None
    ) -> Reply:
        reply = self.client.complete(system, messages, role, schema)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"key": prompt_key(system, messages, role, schema), **_recorded(reply)}
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        return reply


class ReplayClient:
    """Answers from a recording; a prompt it has not seen raises ``ReplayMiss``."""

    def __init__(self, config: LLMConfig, path: Path):
        self.config = config
        self.usage = Usage()
        self.replies = {}
        for line in path.read_text().splitlines():
            record = json.loads(line)
            self.replies[record.pop("key")] = record

    def complete(
        self, system: str, messages: list[dict], role: str | None = None, schema: dict | None = None
    ) -> Reply:
        key = prompt_key(system, messages, role, schema)
        if key not in self.replies:
            raise ReplayMiss(f"no recorded reply for prompt {key}")
        record = self.replies[key]
        reply = Reply(
            text=record["text"],
            content=[{"type": "text", "text": record["text"]}],
            model=record["model"],
            stop_reason=record["stop_reason"],
            usage=Usage(**record["usage"]),
        )
        self.usage.add(reply.usage)
        return reply


def create_client(
    config: LLMConfig,
    transcripts: Path | None = None,
    record: Path | None = None,
    replay: Path | None = None,
) -> Completer:
    if replay is not None:
        return ReplayClient(config, replay)
    if config.backend == "claude-code":
        from smforge.llm.claude_code import ClaudeCodeClient

        client: Completer = ClaudeCodeClient(config, transcripts)
    elif config.backend == "api":
        client = LLMClient(config, transcripts)
    else:
        raise ValueError(f"unknown LLM backend {config.backend!r}; expected api or claude-code")
    return Recorder(client, record) if record is not None else client


def prompt_key(
    system: str, messages: list[dict], role: str | None = None, schema: dict | None = None
) -> str:
    """A hash of the prompt's text, the same whether replies carry SDK blocks or plain text.

    Decimal numbers and rates are masked, so a prompt that quotes timings maps to the same key
    when the measurements differ slightly from the recorded run.
    """
    turns = [{"role": m["role"], "text": _MEASURED.sub("#", _text(m["content"]))} for m in messages]
    body = json.dumps(
        {"system": system, "messages": turns, "role": role, "schema": schema}, sort_keys=True
    )
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
        if text is not None:
            parts.append(text)
    return "".join(parts)


def _recorded(reply: Reply) -> dict:
    return {
        "text": reply.text,
        "model": reply.model,
        "stop_reason": reply.stop_reason,
        "usage": reply.usage.model_dump(),
    }


def check_budget(config: LLMConfig, usage: Usage) -> None:
    if config.budget_usd is not None and usage.cost_usd >= config.budget_usd:
        raise BudgetExceeded(f"spent ${usage.cost_usd:.2f} of the ${config.budget_usd:.2f} budget")


def record_transcript(directory: Path | None, request: dict, reply: Reply) -> None:
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    body = json.dumps(_jsonable(request), sort_keys=True)
    key = hashlib.sha256(body.encode()).hexdigest()[:12]
    path = directory / f"{time.strftime('%Y%m%d-%H%M%S')}-{key}.json"
    record = {
        "request": _jsonable(request),
        "reply": {**reply.model_dump(mode="json", exclude={"content"}), "text": reply.text},
    }
    path.write_text(json.dumps(record, indent=1))


def _usage(usage: Any, served_by: str, requested: str) -> Usage:
    counts = Usage(
        calls=1,
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
    )
    price = PRICES.get(served_by) or PRICES.get(requested)
    if price:
        tokens = (
            counts.input_tokens,
            counts.output_tokens,
            counts.cache_write_tokens,
            counts.cache_read_tokens,
        )
        counts.cost_usd = sum(n * p for n, p in zip(tokens, price, strict=True)) / 1e6
    return counts


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value
