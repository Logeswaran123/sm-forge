import pytest

from smforge.harness.static import validate_source

GOOD = """
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3)],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(a, b, c, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc = tl.dot(tl.load(a), tl.load(b), acc)


class Model(torch.nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        matmul_kernel[(1,)](x, x, out, 1, 1, 1)
        return out
"""


def checks(issues):
    return sorted(issue.check for issue in issues)


def test_good_kernel_passes():
    assert validate_source(GOOD) == []


def test_syntax_error():
    issues = validate_source("def broken(:\n")
    assert checks(issues) == ["syntax_error"]
    assert issues[0].line == 1


def test_missing_structure():
    assert checks(validate_source("x = 1\n")) == [
        "missing_model",
        "missing_triton_import",
        "missing_triton_kernel",
    ]


def test_launch_parameters():
    code = GOOD.replace("num_warps=8", "num_warps=6").replace('"BLOCK_N": 128', '"BLOCK_N": 96')
    code = code.replace('"BLOCK_K": 64', '"BLOCK_K": 8').replace("num_stages=3", "num_stages=0")
    assert checks(validate_source(code)) == ["block_pow2", "dot_min_dim", "num_stages", "num_warps"]


def test_tuple_assigned_blocks():
    code = GOOD + "\nBLOCK_M, BLOCK_N, BLOCK_K = 64, 48, 8\n"
    assert checks(validate_source(code)) == ["block_pow2", "dot_min_dim"]


def test_arange_length():
    code = GOOD.replace("tl.arange(0, BLOCK_M)", "tl.arange(0, 48)")
    issues = validate_source(code)
    assert checks(issues) == ["arange_pow2"]
    assert issues[0].line == 13


def test_shared_memory_estimate():
    assert validate_source(GOOD, smem_limit=200 * 1024, elem_bytes=2) == []
    issues = validate_source(GOOD, smem_limit=64 * 1024, elem_bytes=2)
    assert checks(issues) == ["smem_estimate"]
    assert issues[0].severity == "warning"
    assert "about 96 KB" in issues[0].message


def test_host_sync_in_forward_path():
    code = GOOD.replace(
        "        return out",
        "        self._check(out)\n        return out\n\n    def _check(self, out):\n"
        "        torch.cuda.synchronize()\n        return out.sum().item()",
    )
    issues = validate_source(code)
    assert checks(issues) == ["host_sync", "host_sync"]
    assert all(issue.severity == "warning" for issue in issues)


def test_low_precision_accumulate():
    dot = "tl.dot(tl.load(a), tl.load(b), acc)"
    assert dot in GOOD
    code = GOOD.replace(dot, "acc + tl.dot(tl.load(a), tl.load(b), out_dtype=tl.float16)")
    assert checks(validate_source(code)) == ["low_precision_accumulate"]
    fp32 = GOOD.replace(dot, "tl.dot(tl.load(a), tl.load(b), acc, out_dtype=tl.float32)")
    assert validate_source(fp32) == []


def test_spin_wait():
    code = GOOD.replace(
        "    acc = tl.zeros(",
        "    flag = tl.atomic_add(cnt_ptr, 1)\n"
        "    while flag < 4:\n"
        "        flag = tl.atomic_add(cnt_ptr, 0)\n"
        "    acc = tl.zeros(",
    )
    assert checks(validate_source(code)) == ["spin_wait"]


@pytest.mark.parametrize(
    "line",
    [
        "self._cache = out",
        "self.stats[0] = 1",
        "self.calls += 1",
        "setattr(self, 'packed', out)",
    ],
)
def test_state_in_forward(line):
    code = GOOD.replace("        return out\n", f"        {line}\n        return out\n")
    assert checks(validate_source(code)) == ["state_in_forward"]


def test_state_in_a_helper():
    code = GOOD.replace(
        "        return out\n",
        "        self._keep(out)\n        return out\n\n    def _keep(self, out):\n        self.out = out\n",
    )
    assert checks(validate_source(code)) == ["state_in_forward"]
