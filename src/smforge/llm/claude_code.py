"""LLM calls through headless Claude Code (``claude -p``), on its own login instead of an API key.

A conversation is a Claude Code session: the first message starts one, and later messages resume
it by id, sending only the newest user turn. Tools are off and the system prompt replaces Claude
Code's own. Calls run in an empty temporary directory, so no project instructions are picked up.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from smforge.config import LLMConfig
from smforge.llm.client import Reply, Usage, check_budget, record_transcript

TIMEOUT = 1800.0


class ClaudeCodeError(RuntimeError):
    pass


class ClaudeCodeClient:
    def __init__(self, config: LLMConfig, transcripts: Path | None = None):
        self.config = config
        self.transcripts = transcripts
        self.usage = Usage()
        self._session: str | None = None
        self._workdir = Path(tempfile.mkdtemp(prefix="smforge-claude-"))

    @property
    def binary(self) -> str:
        found = self.config.claude_bin or shutil.which("claude")
        if found is None:
            raise ClaudeCodeError("claude not found; install Claude Code or set CLAUDE_BIN")
        return found

    def complete(self, system: str, messages: list[dict], role: str | None = None) -> Reply:
        check_budget(self.config, self.usage)
        if len(messages) == 1:
            self._session = None
        command = [
            self.binary,
            "-p",
            "--output-format",
            "json",
            "--tools",
            "",
            "--strict-mcp-config",
            "--model",
            self.config.model,
            "--effort",
            self.config.effort_for(role),
            "--system-prompt",
            system,
        ]
        if self._session is not None:
            command += ["--resume", self._session]
        prompt = messages[-1]["content"]
        out = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            cwd=self._workdir,
        )
        try:
            data = json.loads(out.stdout)
        except json.JSONDecodeError as e:
            detail = (out.stderr or out.stdout).strip()[-2000:]
            raise ClaudeCodeError(f"claude exited with code {out.returncode}: {detail}") from e
        if data.get("is_error"):
            raise ClaudeCodeError(str(data.get("result", "")).strip())
        self._session = data["session_id"]
        usage = data.get("usage", {})
        reply = Reply(
            text=data.get("result", ""),
            content=[{"type": "text", "text": data.get("result", "")}],
            model=self.config.model,
            stop_reason=data.get("stop_reason"),
            usage=Usage(
                calls=1,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
                cache_read_tokens=usage.get("cache_read_input_tokens", 0),
                cost_usd=data.get("total_cost_usd", 0.0),
            ),
        )
        self.usage.add(reply.usage)
        request = {"backend": "claude-code", "session": self._session, "system": system}
        record_transcript(self.transcripts, {**request, "prompt": prompt}, reply)
        return reply
