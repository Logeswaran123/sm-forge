"""CoVeR: generate a candidate, verify it on the GPU, feed the failure back, repeat.

The exchange is one conversation: each failed attempt and its verification feedback stay in
context. With ``keep`` set, only the task and the last ``keep`` attempts are sent; older attempts
are dropped from the front.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from smforge.llm.client import Completer, Usage

_CODE_BLOCK = re.compile(r"```(?:python|py)\n(.*?)```", re.DOTALL)
NO_CODE = "The reply has no ```python block. Reply with the complete file in one ```python block."


@dataclass
class Attempt:
    code: str | None
    passed: bool
    feedback: str
    summary: str = ""


@dataclass
class CoverResult:
    code: str | None
    attempts: list[Attempt] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    @property
    def passed(self) -> bool:
        return self.code is not None


Verifier = Callable[[str, int], tuple[bool, str]]


def extract_code(text: str) -> str | None:
    blocks = _CODE_BLOCK.findall(text)
    return blocks[-1] if blocks else None


def extract_summary(text: str, limit: int = 300) -> str:
    """The first line of the reply outside its code blocks."""
    prose = _CODE_BLOCK.sub("", text).strip()
    line = prose.splitlines()[0].strip() if prose else ""
    return line if len(line) <= limit else line[: limit - 3] + "..."


def cover(
    llm: Completer,
    system: str,
    task: str,
    verify: Verifier,
    max_iters: int,
    role: str | None = None,
    keep: int | None = None,
) -> CoverResult:
    """Up to ``max_iters`` attempts; ``verify(code, attempt)`` returns ``(passed, feedback)``."""
    result = CoverResult(code=None)
    messages: list[dict] = [{"role": "user", "content": task}]
    for attempt in range(max_iters):
        if keep is not None and len(messages) > 1 + 2 * keep:
            messages = [messages[0], *messages[len(messages) - 2 * keep :]]
        reply = llm.complete(system, messages, role)
        result.usage.add(reply.usage)
        code = extract_code(reply.text)
        passed, feedback = verify(code, attempt) if code is not None else (False, NO_CODE)
        result.attempts.append(Attempt(code, passed, feedback, extract_summary(reply.text)))
        if passed:
            result.code = code
            return result
        messages += [
            {"role": "assistant", "content": reply.content},
            {
                "role": "user",
                "content": f"Verification failed:\n\n{feedback}\n\n"
                "Fix the kernel and reply with the complete file.",
            },
        ]
    return result
