"""The Anthropic client: model, effort, caching, refusals, usage, cost and transcripts.

The system prompt carries an explicit cache breakpoint and the request enables automatic
caching, so a multi-turn exchange re-reads its history from the cache. A classifier decline is
retried server-side on the model Anthropic recommends for that category (``fallbacks:
"default"``); a decline that survives that raises ``Refused``.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import anthropic
from pydantic import BaseModel

from smforge.config import LLMConfig

FALLBACK_BETA = "server-side-fallback-2026-07-01"
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


class Refused(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


class LLMClient:
    def __init__(self, config: LLMConfig, transcripts: Path | None = None):
        self.config = config
        self.transcripts = transcripts
        self.usage = Usage()
        self._client: anthropic.Anthropic | None = None

    def complete(self, system: str, messages: list[dict]) -> Reply:
        if self.config.budget_usd is not None and self.usage.cost_usd >= self.config.budget_usd:
            raise BudgetExceeded(
                f"spent ${self.usage.cost_usd:.2f} of the ${self.config.budget_usd:.2f} budget"
            )
        request = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
            "cache_control": {"type": "ephemeral"},
            "output_config": {"effort": self.config.effort},
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
        self._record(request, reply)
        if reply.stop_reason == "refusal":
            raise Refused(f"{message.model} declined: {message.stop_details}")
        return reply

    def _send(self, request: dict) -> Any:
        if self._client is None:
            self._client = anthropic.Anthropic()
        with self._client.beta.messages.stream(**request) as stream:
            return stream.get_final_message()

    def _record(self, request: dict, reply: Reply) -> None:
        if self.transcripts is None:
            return
        self.transcripts.mkdir(parents=True, exist_ok=True)
        body = json.dumps(_jsonable(request), sort_keys=True)
        key = hashlib.sha256(body.encode()).hexdigest()[:12]
        path = self.transcripts / f"{time.strftime('%Y%m%d-%H%M%S')}-{key}.json"
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
