"""Source-level view of a kernel file: imports, Triton functions, the Model class."""

from __future__ import annotations

import ast
from functools import cached_property

_TRITON_DECORATORS = frozenset({"jit", "autotune", "heuristics"})


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
