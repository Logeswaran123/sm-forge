"""Static checks on the CUDA and C++ sources of a ``cuda`` kernel file. Text analysis only.

Errors: a file that launches kernels with ``<<<...>>>`` but never calls ``cudaGetLastError``,
``cudaPeekAtLastError`` or ``C10_CUDA_KERNEL_LAUNCH_CHECK`` (a launch that fails leaves the
output as ``torch.empty`` found it, which can hold an earlier correct result); ``cudaMalloc`` or
``cudaFree`` per call (both synchronize the device; allocate with ``torch::empty`` in the
binding); a ``static`` variable, which keeps state between calls; a PyTorch header in the CUDA
source. Warnings: a ``__global__`` kernel without ``__launch_bounds__`` (ptxas does not know the
block size, so it may use more registers than a large block can launch with, and cannot trade
registers for resident blocks), or without a comparison against one of its integer parameters
(no bounds check), a launch on the default stream, and a host synchronization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from smforge.cuda import TORCH_HEADER
from smforge.harness.source import CudaSource, KernelSource
from smforge.models import Severity, ValidationIssue

_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_STRING = re.compile(r'"(?:\\.|[^"\\\n])*"')
_KERNEL = re.compile(
    r"__global__\s+(?:void\s+)?(?:__launch_bounds__\s*\((?P<bounds>[^)]*)\)\s+)?(?:void\s+)?"
    r"(?P<name>\w+)\s*\("
)
_LAUNCH = re.compile(r"<<<(?P<config>.*?)>>>", re.S)
_LAUNCH_CHECK = re.compile(
    r"\b(?:cudaGetLastError|cudaPeekAtLastError|C10_CUDA_KERNEL_LAUNCH_CHECK)\b"
)
_ALLOC = re.compile(r"\bcuda(?:Malloc|MallocManaged|MallocHost|Free|FreeHost)\s*\(")
_SYNC = re.compile(
    r"\bcuda(?:DeviceSynchronize|StreamSynchronize|EventSynchronize|ThreadSynchronize|Memcpy|Memcpy2D)"
    r"\s*\(|\btorch::cuda::synchronize\b|\.(?:item|cpu|tolist)\s*(?:<[^>()]*>)?\s*\("
)
_STATIC = re.compile(
    r"\bstatic\s+(?!constexpr\b|const\b|inline\b|__\w+|void\b)[\w:<>,*&\s]*?\b\w+\s*(?:=|;|\[|\{)"
)
_SCALAR = re.compile(
    r"\b(?:const\s+)?(?:unsigned\s+)?(?:int|long|long\s+long|int64_t|int32_t|uint32_t|uint64_t|size_t|index_t)\s+(\w+)\s*$"
)
_CONDITION = re.compile(r"\b(?:if|for|while)\s*\(")


@dataclass
class Kernel:
    name: str
    bounds: str | None
    params: str
    body: str
    start: int


def check(src: KernelSource) -> list[ValidationIssue]:
    issues = []
    for extension in src.extensions.values():
        texts = {source.kind: code_only(source.text) for source in extension.sources}
        launches_checked = any(_LAUNCH_CHECK.search(text) for text in texts.values())
        for source in extension.sources:
            code = texts[source.kind]
            issues += _patterns(source, code)
            if source.kind != "cuda":
                continue
            if found := TORCH_HEADER.search(source.text):
                issues.append(
                    _issue(
                        "torch_header",
                        "the CUDA source includes a PyTorch header (about 20 s per build); "
                        "take raw pointers and a cudaStream_t, and unpack tensors in the binding",
                        source.file_line(found.start()),
                    )
                )
            launches = list(_LAUNCH.finditer(code))
            if launches and not launches_checked:
                issues.append(
                    _issue(
                        "launch_check",
                        "kernels are launched but no cudaGetLastError or "
                        "C10_CUDA_KERNEL_LAUNCH_CHECK follows; a failed launch leaves the "
                        "output unwritten",
                        source.file_line(launches[0].start()),
                    )
                )
            for launch in launches:
                if len(_split(launch.group("config"))) < 4:
                    issues.append(
                        _issue(
                            "launch_stream",
                            "the launch uses the default stream; pass the stream from "
                            "at::cuda::getCurrentCUDAStream()",
                            source.file_line(launch.start()),
                            Severity.WARNING,
                        )
                    )
            for kernel in kernels(code):
                line = source.file_line(kernel.start)
                if kernel.bounds is None:
                    issues.append(
                        _issue(
                            "launch_bounds",
                            f"{kernel.name} has no __launch_bounds__; ptxas does not know its "
                            "block size, so it may use more registers than the block can launch "
                            "with",
                            line,
                            Severity.WARNING,
                        )
                    )
                if not _bounds_checked(kernel):
                    issues.append(
                        _issue(
                            "bounds_check",
                            f"{kernel.name} never compares an index with its size parameters; "
                            "out-of-range threads read and write past the tensors",
                            line,
                            Severity.WARNING,
                        )
                    )
    return issues


def kernels(code: str) -> list[Kernel]:
    """``__global__`` functions in ``code`` (comments and strings removed)."""
    found = []
    for match in _KERNEL.finditer(code):
        open_paren = match.end() - 1
        close_paren = _matching(code, open_paren, "(", ")")
        brace = code.find("{", close_paren)
        if close_paren < 0 or brace < 0 or ";" in code[close_paren:brace]:
            continue
        end = _matching(code, brace, "{", "}")
        found.append(
            Kernel(
                match.group("name"),
                match.group("bounds"),
                code[open_paren + 1 : close_paren],
                code[brace + 1 : end if end > 0 else len(code)],
                match.start(),
            )
        )
    return found


def _patterns(source: CudaSource, code: str) -> list[ValidationIssue]:
    issues = []
    for match in _ALLOC.finditer(code):
        issues.append(
            _issue(
                "device_alloc",
                f"{match.group(0).rstrip('( ')} on every call synchronizes the device; allocate "
                "scratch with torch::empty in the binding",
                source.file_line(match.start()),
            )
        )
    for match in _STATIC.finditer(code):
        issues.append(
            _issue(
                "static_state",
                "a static variable keeps state between calls; recompute it on every call",
                source.file_line(match.start()),
            )
        )
    for match in _SYNC.finditer(code):
        issues.append(
            _issue(
                "host_sync",
                f"{match.group(0).rstrip('( ')} waits for the GPU",
                source.file_line(match.start()),
                Severity.WARNING,
            )
        )
    return issues


def _bounds_checked(kernel: Kernel) -> bool:
    scalars = set()
    for param in _split(kernel.params):
        if match := _SCALAR.search(param.strip()):
            scalars.add(match.group(1))
    if not scalars:
        return True
    for match in _CONDITION.finditer(kernel.body):
        end = _matching(kernel.body, match.end() - 1, "(", ")")
        condition = kernel.body[match.end() : end]
        if re.search(r"[<>]", condition) and any(
            re.search(rf"\b{re.escape(name)}\b", condition) for name in scalars
        ):
            return True
    return any(
        re.search(rf"\b(?:min|max)\w*\s*\([^;]*\b{re.escape(n)}\b", kernel.body) for n in scalars
    )


def code_only(text: str) -> str:
    """``text`` with comments and string literals blanked, offsets and newlines kept."""

    def blank(match: re.Match) -> str:
        return re.sub(r"[^\n]", " ", match.group(0))

    return _STRING.sub(blank, _COMMENT.sub(blank, text))


def _matching(text: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    for i in range(start, len(text)):
        if text[i] == opening:
            depth += 1
        elif text[i] == closing:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _split(text: str) -> list[str]:
    """Top-level comma-separated parts of ``text``."""
    parts, depth, current = [], 0, ""
    for char in text:
        if char in "(<[{":
            depth += 1
        elif char in ")>]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current)
    return parts


def _issue(
    check: str, message: str, line: int, severity: Severity = Severity.ERROR
) -> ValidationIssue:
    return ValidationIssue(check=check, severity=severity, message=message, line=line)
