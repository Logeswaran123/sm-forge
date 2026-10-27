"""CoVeR: generate a candidate, verify it on the GPU, feed the failure back, repeat.

The exchange is one growing conversation: each failed attempt and its verification feedback stay
in context, and the history is only ever appended to.
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


def cover(llm: Completer, system: str, task: str, verify: Verifier, max_iters: int) -> CoverResult:
    """Up to ``max_iters`` attempts; ``verify(code, attempt)`` returns ``(passed, feedback)``."""
    result = CoverResult(code=None)
    messages: list[dict] = [{"role": "user", "content": task}]
    for attempt in range(max_iters):
        reply = llm.complete(system, messages)
        result.usage.add(reply.usage)
        code = extract_code(reply.text)
        passed, feedback = verify(code, attempt) if code is not None else (False, NO_CODE)
        result.attempts.append(Attempt(code=code, passed=passed, feedback=feedback))
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
