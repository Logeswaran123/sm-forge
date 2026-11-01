"""Building CUDA C++ kernels as PyTorch extensions, for kernel files in the ``cuda`` DSL.

A ``cuda`` kernel file keeps its device code and its PyTorch binding in two strings and builds
them when it is imported::

    ext = build(CUDA_SRC, CPP_SRC, ["forward"])

``CUDA_SRC`` is compiled by nvcc alone and includes no PyTorch header: its host functions launch
the kernels and take raw pointers, sizes and a ``cudaStream_t``. ``CPP_SRC`` declares those
functions, unpacks the tensors and calls them, and is compiled against a precompiled
``torch/extension.h``. PyTorch's headers take about 20 s to compile, so with them kept out of
nvcc and precompiled for the host compiler, a build takes a few seconds.

Builds are cached under ``<cache>/cuda/`` by a hash of the sources, flags, GPU and PyTorch build,
so a kernel file loads at once after its first build, in any process. A source that includes
``cutlass/`` gets the CUTLASS headers, fetched once at ``CUTLASS_VERSION`` (or ``CUTLASS_DIR``).
A failed build raises ``BuildError`` with the compiler's diagnostics mapped to lines of the
kernel file.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

CUTLASS_VERSION = "v4.8.1"
CUTLASS_REPO = "https://github.com/NVIDIA/cutlass.git"
CUTLASS_PATHS = ("include", "tools/util/include", "examples/41_fused_multi_head_attention")
NVCC_FLAGS = (
    "-O3",
    "-std=c++17",
    "-lineinfo",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    # PyTorch hides the CUDA half and bfloat16 operators for its own headers, which the device
    # code does not include.
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
)
BINDING_HEADERS = ("torch/extension.h", "ATen/cuda/CUDAContext.h", "c10/cuda/CUDAStream.h")
TORCH_HEADER = re.compile(r'^\s*#\s*include\s*[<"](?:torch|ATen|c10)/', re.M)
_NVCC = re.compile(
    r"^(?P<file>\S.*?)\((?P<line>\d+)\): (?P<kind>[a-z ]*error|warning)\S*: (?P<text>.*)"
)
_GCC = re.compile(
    r"^(?P<file>[^\s:]+):(?P<line>\d+):(?:\d+:)? (?P<kind>fatal error|error): (?P<text>.*)"
)
_PTXAS = re.compile(r"^ptxas (?:fatal|error)\s*: (?P<text>.*)")
_INSTANTIATED = re.compile(r"at line (?P<line>\d+) of (?P<file>\S+)")
_UNDEFINED = re.compile(r"undefined symbol: (\S+)")
_loaded: dict[str, Path] = {}


@dataclass
class Diagnostic:
    """A compiler error. ``line`` is in the kernel file, when the source string is found there."""

    source: str
    line: int | None
    text: str

    def __str__(self) -> str:
        where = f"{self.source} source" if self.line is None else f"line {self.line}"
        return f"{where}: {self.text}"


class BuildError(RuntimeError):
    def __init__(self, diagnostics: list[Diagnostic], log: str):
        self.diagnostics = diagnostics
        self.log = log
        if diagnostics:
            shown = diagnostics[:8]
            text = "\n".join(map(str, shown))
            if len(diagnostics) > len(shown):
                text += f"\n... {len(diagnostics) - len(shown)} more"
        else:
            text = log.strip()[-3000:]
        super().__init__(f"CUDA build failed:\n{text}")


def build(cuda: str, cpp: str, functions: Sequence[str], flags: Sequence[str] = ()) -> ModuleType:
    """The extension built from ``cuda`` and ``cpp``, exposing ``functions``; cached."""
    import torch
    from torch.utils import cpp_extension

    caller = _read(Path(inspect.stack()[1].filename))
    if found := TORCH_HEADER.search(cuda):
        line = _file_line(cuda, cuda[: found.start()].count("\n") + 1, caller)
        text = (
            "the CUDA source includes a PyTorch header, which adds about 20 s to every build; "
            "take raw pointers, sizes and a cudaStream_t, and unpack tensors in the C++ binding"
        )
        raise BuildError([Diagnostic("cuda", line, text)], "")
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}" + ("a" if major >= 9 else "")
    includes = []
    if "cutlass/" in cuda or "cutlass/" in cpp:
        root = cutlass_dir()
        includes = [str(root / "include"), str(root / "tools" / "util" / "include")]
    nvcc = [*NVCC_FLAGS, f"-arch={arch}", *flags]
    key = "\0".join([cuda, cpp, *functions, *nvcc, *includes, torch.__version__, arch])
    name = f"smf_{hashlib.sha256(key.encode()).hexdigest()[:16]}"
    directory = cache_root() / name
    directory.mkdir(parents=True, exist_ok=True)
    header = binding_header()
    _ninja_on_path()
    try:
        module = cpp_extension.load_inline(
            name,
            cpp,
            cuda,
            functions=list(functions),
            extra_cflags=["-include", str(header), "-Winvalid-pch"],
            extra_cuda_cflags=nvcc,
            extra_include_paths=includes,
            build_directory=str(directory),
            no_implicit_headers=True,
        )
    except ImportError as e:
        text = str(e)
        if match := _UNDEFINED.search(text):
            text = f"undefined symbol {_demangle(match.group(1))}: declared but not defined"
        raise BuildError([Diagnostic("cpp", None, text)], str(e)) from None
    except RuntimeError as e:
        raise BuildError(parse_log(str(e), cuda, cpp, caller), str(e)) from None
    _loaded[name] = directory / f"{name}.so"
    return module


def function_name(demangled: str) -> str:
    """``void k<float>(float const*, int)`` -> ``k<float>``: no return type, no parameters."""
    depth = 0
    for i, char in enumerate(demangled):
        depth += char == "<"
        depth -= char == ">"
        if char == "(" and depth == 0:
            return demangled[:i].removeprefix("void ")
    return demangled.removeprefix("void ")


def loaded() -> dict[str, Path]:
    """Extension name to shared library, for every extension built or loaded in this process."""
    return dict(_loaded)


def parse_log(log: str, cuda: str, cpp: str, kernel_file: str | None = None) -> list[Diagnostic]:
    """Errors in a build log; lines in ``cuda.cu`` and ``main.cpp`` map to the kernel file."""
    diagnostics = []
    lines = log.splitlines()
    for i, line in enumerate(lines):
        line = line.strip()
        match = _NVCC.match(line) or _GCC.match(line)
        if match and "warning" not in match.group("kind"):
            file, number = Path(match.group("file")).name, int(match.group("line"))
            text = match.group("text").strip()
            if file not in ("cuda.cu", "main.cpp"):
                text = f"{text} (in {_short(match.group('file'))}:{number})"
                for later in lines[i + 1 : i + 40]:
                    if (found := _INSTANTIATED.search(later)) and found.group("file").endswith(
                        "cuda.cu"
                    ):
                        file, number = "cuda.cu", int(found.group("line"))
                        break
                    if _NVCC.match(later.strip()):
                        break
            source = "cpp" if file == "main.cpp" else "cuda"
            text_of = cpp if source == "cpp" else cuda
            mapped = None
            if file in ("cuda.cu", "main.cpp") and number <= text_of.count("\n") + 1:
                mapped = _file_line(text_of, number, kernel_file)
            diagnostics.append(Diagnostic(source, mapped, text))
        elif found := _PTXAS.match(line):
            diagnostics.append(Diagnostic("cuda", None, f"ptxas: {found.group('text')}"))
    return diagnostics


def cache_root() -> Path:
    from smforge.config import get_config

    return (get_config().cache_dir / "cuda").resolve()


def binding_header() -> Path:
    """A header including ``BINDING_HEADERS``, precompiled with the flags the binding builds with.

    Include paths do not invalidate a GCC precompiled header, so one serves every build.
    """
    import torch
    from torch.utils import cpp_extension

    compiler = cpp_extension.get_cxx_compiler()
    command = [
        compiler,
        "-x",
        "c++-header",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        *(
            flag
            for path in [
                *cpp_extension.include_paths(device_type="cuda"),
                sysconfig.get_path("include"),
            ]
            for flag in ("-isystem", path)
        ),
        "-fPIC",
        "-std=c++17",
    ]
    key = hashlib.sha256("\0".join([*command, torch.__version__]).encode()).hexdigest()[:16]
    directory = cache_root() / "pch" / key
    header = directory / "binding.h"
    if (directory / "binding.h.gch").exists():
        return header
    directory.mkdir(parents=True, exist_ok=True)
    header.write_text("".join(f"#include <{h}>\n" for h in BINDING_HEADERS))
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".gch", delete=False) as f:
        partial = Path(f.name)
    try:
        subprocess.run([*command, str(header), "-o", str(partial)], check=True, capture_output=True)
        partial.replace(directory / "binding.h.gch")
    except subprocess.CalledProcessError as e:
        raise BuildError(
            [], f"precompiling the binding header failed:\n{e.stderr.decode()}"
        ) from None
    finally:
        partial.unlink(missing_ok=True)
    return header


def cutlass_dir() -> Path:
    """CUTLASS headers: ``CUTLASS_DIR``, else a sparse checkout of ``CUTLASS_VERSION``."""
    if env := os.environ.get("CUTLASS_DIR"):
        return Path(env)
    target = cache_root().parent / "cutlass" / CUTLASS_VERSION
    if (target / "include" / "cutlass" / "cutlass.h").exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(dir=target.parent, prefix=f"{CUTLASS_VERSION}."))
    git = shutil.which("git")
    if git is None:
        raise BuildError([], "git not found: install it, or set CUTLASS_DIR to a CUTLASS checkout")
    try:
        subprocess.run(
            [
                *(git, "clone", "-q", "--depth", "1", "--branch", CUTLASS_VERSION),
                *("--filter=blob:none", "--sparse", CUTLASS_REPO, str(partial)),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [git, "-C", str(partial), "sparse-checkout", "set", *CUTLASS_PATHS],
            check=True,
            capture_output=True,
        )
        if not target.exists():
            partial.rename(target)
    except subprocess.CalledProcessError as e:
        raise BuildError(
            [], f"fetching CUTLASS {CUTLASS_VERSION} failed:\n{e.stderr.decode()}"
        ) from None
    finally:
        if partial.exists():
            shutil.rmtree(partial)
    return target


def _ninja_on_path() -> None:
    """PyTorch runs ``ninja`` from PATH; a venv over a conda env may hold it only in the base."""
    if shutil.which("ninja"):
        return
    candidates = [Path(sys.base_prefix) / "bin", Path(sysconfig.get_path("scripts"))]
    try:
        import ninja

        candidates.insert(0, Path(ninja.BIN_DIR))
    except ImportError:
        pass
    for directory in candidates:
        if (directory / "ninja").exists():
            os.environ["PATH"] = f"{directory}{os.pathsep}{os.environ.get('PATH', '')}"
            return


def _file_line(source: str, line: int, kernel_file: str | None) -> int | None:
    """Line ``line`` of ``source`` as a line of the kernel file that contains it."""
    if kernel_file is None or (start := kernel_file.find(source)) < 0:
        return None
    return kernel_file[:start].count("\n") + line


def _read(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


def _short(path: str) -> str:
    for marker in ("/include/", "/cuda.cu", "/main.cpp"):
        if marker in path:
            return path[path.index(marker) + 1 :]
    return Path(path).name


def _demangle(symbol: str) -> str:
    tool = shutil.which("c++filt")
    if tool is None:
        return symbol
    out = subprocess.run([tool, symbol], capture_output=True, text=True)
    return out.stdout.strip() or symbol
