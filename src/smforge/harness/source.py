"""Source-level view of a kernel file: imports, Triton functions, CUDA sources, the Model class."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import cached_property

from smforge.models import Dsl

_TRITON_DECORATORS = frozenset({"jit", "autotune", "heuristics"})
BUILD = "smforge.cuda.build"


@dataclass(frozen=True)
class CudaSource:
    """A source string passed to ``smforge.cuda.build``; ``kind`` is ``cuda`` or ``cpp``."""

    kind: str
    text: str
    line: int

    def file_line(self, offset: int) -> int:
        """The kernel-file line of character ``offset`` in ``text``."""
        return self.line + self.text.count("\n", 0, offset)


@dataclass(frozen=True)
class Extension:
    """``name = build(cuda, cpp, functions)`` at module level."""

    name: str
    line: int
    sources: tuple[CudaSource, ...]
    functions: tuple[str, ...]


class KernelSource:
    def __init__(self, code: str):
        self.code = code
        self.tree = ast.parse(code)

    @cached_property
    def aliases(self) -> dict[str, str]:
        """Local name to the dotted path it was imported as."""
        aliases = {}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        aliases[alias.asname] = alias.name
                    else:
                        root = alias.name.split(".")[0]
                        aliases[root] = root
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                for alias in node.names:
                    aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        return aliases

    def dotted(self, node: ast.expr) -> str | None:
        """``F.relu`` -> ``torch.nn.functional.relu`` when ``F`` is an import alias."""
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name) or node.id not in self.aliases:
            return None
        parts.append(self.aliases[node.id])
        return ".".join(reversed(parts))

    def calls(self, node: ast.AST) -> list[tuple[ast.Call, str | None]]:
        return [(n, self.dotted(n.func)) for n in ast.walk(node) if isinstance(n, ast.Call)]

    @cached_property
    def dsl(self) -> Dsl:
        cuda = any(
            p == "smforge.cuda" or p.startswith("smforge.cuda.") for p in self.aliases.values()
        )
        return Dsl.CUDA if cuda else Dsl.TRITON

    @cached_property
    def extensions(self) -> dict[str, Extension]:
        """Module-level ``smforge.cuda.build`` calls whose sources are string constants."""
        strings = {}
        for node in self.tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                strings[node.targets[0].id] = node.value
        found = {}
        for node in self.tree.body:
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and self.dotted(node.value.func) == BUILD
            ):
                continue
            call = node.value
            args = dict(zip(("cuda", "cpp", "functions"), call.args, strict=False))
            args |= {k.arg: k.value for k in call.keywords if k.arg}
            sources = []
            for kind in ("cuda", "cpp"):
                value = args.get(kind)
                if isinstance(value, ast.Name):
                    value = strings.get(value.id)
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    sources.append(CudaSource(kind, value.value, value.lineno))
            names = args.get("functions")
            functions = tuple(
                e.value
                for e in getattr(names, "elts", [])
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            )
            name = node.targets[0].id
            found[name] = Extension(name, node.lineno, tuple(sources), functions)
        return found

    def cuda_sources(self, kind: str | None = None) -> list[CudaSource]:
        return [
            source
            for extension in self.extensions.values()
            for source in extension.sources
            if kind is None or source.kind == kind
        ]

    @cached_property
    def functions(self) -> dict[str, ast.FunctionDef]:
        return {n.name: n for n in self.tree.body if isinstance(n, ast.FunctionDef)}

    def is_triton(self, fn: ast.FunctionDef) -> bool:
        for decorator in fn.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            path = self.dotted(target) or ""
            if path.startswith("triton") and path.rsplit(".", 1)[-1] in _TRITON_DECORATORS:
                return True
        return False

    @cached_property
    def triton_functions(self) -> dict[str, ast.FunctionDef]:
        return {name: fn for name, fn in self.functions.items() if self.is_triton(fn)}

    @cached_property
    def model(self) -> ast.ClassDef | None:
        for node in self.tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "Model":
                return node
        return None

    @cached_property
    def methods(self) -> dict[str, ast.FunctionDef]:
        if self.model is None:
            return {}
        return {n.name: n for n in self.model.body if isinstance(n, ast.FunctionDef)}

    def forward_path(self) -> list[ast.FunctionDef]:
        """``Model.forward`` and every module function or method it reaches by direct call."""
        forward = self.methods.get("forward")
        if forward is None:
            return []
        path, queue = [forward], [forward]
        while queue:
            for call in ast.walk(queue.pop()):
                if not isinstance(call, ast.Call):
                    continue
                target = self._callee(call.func)
                if target is not None and not self.is_triton(target) and target not in path:
                    path.append(target)
                    queue.append(target)
        return path

    def _callee(self, func: ast.expr) -> ast.FunctionDef | None:
        if isinstance(func, ast.Name):
            return self.functions.get(func.id)
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
        ):
            return self.methods.get(func.attr)
        return None
