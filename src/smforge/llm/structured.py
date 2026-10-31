"""Replies that must match a pydantic model.

The model's JSON schema is made strict (every object closed and every property required), sent
as the request's output format, and the reply is validated against the model.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from smforge.llm.client import Completer, Reply

T = TypeVar("T", bound=BaseModel)
_DROPPED = frozenset({"title", "default"})


def strict_schema(model: type[BaseModel]) -> dict:
    return _strict(model.model_json_schema())


def ask(
    llm: Completer, system: str, messages: list[dict], model: type[T], role: str | None = None
) -> tuple[T, Reply]:
    reply = llm.complete(system, messages, role, strict_schema(model))
    return model.model_validate_json(reply.text), reply


def _strict(node: Any, names: bool = False) -> Any:
    """``names`` marks a mapping from property or definition names to schemas."""
    if isinstance(node, list):
        return [_strict(item) for item in node]
    if not isinstance(node, dict):
        return node
    if names:
        return {k: _strict(v) for k, v in node.items()}
    out = {
        k: _strict(v, names=k in ("properties", "$defs"))
        for k, v in node.items()
        if k not in _DROPPED
    }
    if out.get("type") == "object" and "properties" in out:
        out["additionalProperties"] = False
        out["required"] = list(out["properties"])
    return out
