"""Translate a KernelBench problem into a spec.

KernelBench declares shapes as module-level constants used by ``get_inputs`` and
``get_init_inputs``. Every int constant those functions use becomes a dim, so a variant can change
it; tuples keep their link to the dims they were built from.

``translate`` reads the source only: constants are evaluated from the syntax tree, and nothing in
the file runs. ``verify`` imports the module and checks the translation against it (on the meta
device, ``get_inputs`` must produce the spec's shapes and dtypes, and ``get_init_inputs()`` must
equal the spec's init arguments), so it runs in the harness worker.
"""

from __future__ import annotations

import ast
import operator
import re
from pathlib import Path
from typing import Any

import torch

from smforge.harness.workload import load_module
from smforge.spec import KernelSpec, Variant

LEVEL1 = (
    1,
    2,
    3,
    4,
    6,
    7,
    8,
    9,
    10,
    16,
    17,
    18,
    23,
    24,
    26,
    33,
    35,
    36,
    40,
    47,
    48,
    49,
    88,
    89,
    95,
    97,
)
LEVEL1_FAMILIES = (
    (range(1, 19), "matmul"),
    (range(19, 33), "activation"),
    (range(33, 41), "norm"),
    (range(41, 47), "pool"),
    (range(47, 54), "reduction"),
    (range(54, 88), "conv"),
    (range(88, 89), "activation"),
    (range(89, 94), "scan"),
    (range(94, 101), "loss"),
)
LEVEL2_FAMILIES = (
    "convtranspose3d",
    "convtranspose2d",
    "conv3d",
    "conv2d",
    "gemm",
    "matmul",
    "bmm",
)
_DTYPES = {
    "float": "float32",
    "float32": "float32",
    "half": "float16",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "long": "int64",
    "int64": "int64",
    "int": "int32",
    "int32": "int32",
    "bool": "bool",
}


_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


class Untranslatable(ValueError):
    pass


def problem_number(path: Path) -> int:
    return int(path.name.split("_", 1)[0])


def problem_id(level: int, path: Path) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", path.stem.split("_", 1)[1].lower()).strip("_")
    return f"l{level}_{problem_number(path):03d}_{slug}"


def family(level: int, path: Path) -> str:
    if level == 1:
        number = problem_number(path)
        return next(name for numbers, name in LEVEL1_FAMILIES if number in numbers)
    first = path.stem.split("_", 2)[1].lower()
    return next((name for name in LEVEL2_FAMILIES if first.startswith(name)), first)


def translate(path: Path) -> tuple[dict, dict[str, int]]:
    """The spec's ``inputs`` and ``inits`` sections, and the dims at KernelBench's values."""
    tree = ast.parse(path.read_text())
    return _Translator(tree, constants(tree.body)).translate()


def constants(body: list[ast.stmt]) -> dict[str, Any]:
    """Module-level names bound to literals, tuples and arithmetic on earlier constants."""
    namespace: dict[str, Any] = {}
    for node in body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = evaluate(node.value, namespace)
        except Untranslatable:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                namespace[target.id] = value
            elif isinstance(target, ast.Tuple) and isinstance(value, tuple):
                for element, item in zip(target.elts, value, strict=False):
                    if isinstance(element, ast.Name):
                        namespace[element.id] = item
    return namespace


def evaluate(node: ast.expr, namespace: dict[str, Any]) -> Any:
    match node:
        case ast.Constant(value=value) if value is None or isinstance(
            value, bool | int | float | str
        ):
            return value
        case ast.Name(id=name) if name in namespace:
            return namespace[name]
        case ast.Tuple(elts=elements) | ast.List(elts=elements):
            return tuple(evaluate(e, namespace) for e in elements)
        case ast.BinOp(left=left, op=op, right=right) if type(op) in _BINOPS:
            return _BINOPS[type(op)](evaluate(left, namespace), evaluate(right, namespace))
        case ast.UnaryOp(op=ast.USub(), operand=operand):
            return -evaluate(operand, namespace)
    raise Untranslatable(f"cannot evaluate {ast.unparse(node)}")


def verify(path: Path, spec: KernelSpec, dims: dict[str, int]) -> None:
    module = load_module(path)
    variant = Variant(params=list(spec.inputs), dims=dims)
    with torch.device("meta"):
        expected = module.get_inputs()
    shapes = [tuple(t.shape) for t in expected]
    dtypes = [str(t.dtype).removeprefix("torch.") for t in expected]
    if shapes != spec.input_shapes(variant) or dtypes != spec.input_dtypes(variant):
        raise Untranslatable(f"inputs {shapes} {dtypes} translated as {spec.input_shapes(variant)}")
    init_args = [tuple(a) if isinstance(a, list) else a for a in module.get_init_inputs()]
    if init_args != spec.init_args(variant):
        raise Untranslatable(f"init args {init_args} translated as {spec.init_args(variant)}")


class _Translator:
    def __init__(self, tree: ast.Module, namespace: dict[str, Any]):
        self.namespace = namespace
        self.functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assignments = _assignments(tree.body)
        self.models = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        self.dims: dict[str, int] = {}

    def translate(self) -> tuple[dict, dict[str, int]]:
        inputs = {name: self._tensor(node) for name, node in self._returned("get_inputs")}
        params = self._init_params()
        inits = []
        for i, (_, node) in enumerate(self._returned("get_init_inputs")):
            inits.append({params[i] if i < len(params) else f"arg{i}": self._symbolic(node)})
        return {"inputs": inputs, "inits": inits}, dict(self.dims)

    def _returned(self, name: str) -> list[tuple[str, ast.expr]]:
        fn = self.functions.get(name)
        if fn is None:
            raise Untranslatable(f"no {name}")
        local = _assignments(fn.body)
        returned = fn.body[-1]
        if not isinstance(returned, ast.Return) or not isinstance(
            returned.value, ast.List | ast.Tuple
        ):
            raise Untranslatable(f"{name} does not return a list literal")
        items = []
        for i, element in enumerate(returned.value.elts):
            if isinstance(element, ast.Name) and element.id in local:
                items.append((element.id, local[element.id]))
            else:
                items.append((element.id if isinstance(element, ast.Name) else f"x{i}", element))
        return items

    def _init_params(self) -> list[str]:
        init = next(
            (n for n in self.models["Model"].body if getattr(n, "name", "") == "__init__"), None
        )
        return [a.arg for a in init.args.args[1:]] if init else []

    def _tensor(self, node: ast.expr) -> dict:
        if not isinstance(node, ast.Call):
            raise Untranslatable(
                f"input {ast.unparse(node)} is not a torch.rand/randn/randint call"
            )
        func = ast.unparse(node.func)
        keywords = {k.arg: k.value for k in node.keywords}
        dtype = self._dtype(keywords.pop("dtype", None))
        if func in ("torch.rand", "torch.randn") and not keywords:
            args = node.args
            if len(args) == 1 and isinstance(args[0], ast.Tuple | ast.List):
                args = args[0].elts
            entry = {"shape": self._shape(args), "dtype": dtype or "float32"}
            if func == "torch.rand":
                entry["range"] = [0.0, 1.0]
            return entry
        if func == "torch.randint":
            args = [*node.args, *(v for k, v in keywords.items() if k == "size")]
            if len(args) != 3 or set(keywords) - {"size"}:
                raise Untranslatable(f"randint form {ast.unparse(node)}")
            low, high, size = args
            dims = size.elts if isinstance(size, ast.Tuple | ast.List) else [size]
            return {
                "shape": self._shape(dims),
                "dtype": dtype or "int64",
                "range": [self._symbolic(low), self._symbolic(high)],
            }
        raise Untranslatable(f"input {ast.unparse(node)}")

    def _shape(self, args: list[ast.expr]) -> list:
        shape = []
        for arg in args:
            if isinstance(arg, ast.Starred):
                expanded = self._symbolic(arg.value)
                if not isinstance(expanded, list):
                    raise Untranslatable(f"*{ast.unparse(arg.value)} is not a tuple")
                shape.extend(expanded)
            else:
                shape.append(self._symbolic(arg))
        if not all(isinstance(d, str | int) and not isinstance(d, bool) for d in shape):
            raise Untranslatable(f"shape {shape}")
        return shape

    def _dtype(self, node: ast.expr | None) -> str | None:
        if node is None:
            return None
        name = ast.unparse(node).removeprefix("torch.")
        if name not in _DTYPES:
            raise Untranslatable(f"dtype {name}")
        return _DTYPES[name]

    def _symbolic(self, node: ast.expr) -> Any:
        """A literal, or the dim name for an int constant; tuples become lists of these."""
        match node:
            case ast.Constant(value=value):
                return value
            case ast.Name(id=name) if name in self.namespace:
                value = self.namespace[name]
                if type(value) is int:
                    self.dims[name.upper()] = value
                    return name.upper()
                source = self.assignments.get(name)
                if isinstance(value, tuple | list) and isinstance(source, ast.Tuple | ast.List):
                    return [self._symbolic(e) for e in source.elts]
                return _plain(value)
            case ast.Tuple(elts=elements) | ast.List(elts=elements):
                return [self._symbolic(e) for e in elements]
        return _plain(evaluate(node, self.namespace))


def _assignments(body: list[ast.stmt]) -> dict[str, ast.expr]:
    return {
        target.id: node.value
        for node in body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    raise Untranslatable(f"value {value!r}")
