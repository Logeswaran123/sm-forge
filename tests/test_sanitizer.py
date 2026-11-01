from pathlib import Path

import pytest

from smforge.harness.sanitizer import SanitizerError, parse, sanitize

PROBLEM = Path(__file__).parent / "kernels" / "linear_relu"

MEMCHECK = """\
========= COMPUTE-SANITIZER
========= Invalid __global__ write of size 2 bytes
=========     at linear_relu_kernel+0x2750 in oob.py:46
=========     by thread (96,0,0) in block (1,0,0)
=========     Address 0x71461beb0 is out of bounds
=========
========= Invalid __global__ write of size 2 bytes
=========     at linear_relu_kernel+0x2750 in oob.py:46
=========     by thread (97,0,0) in block (1,0,0)
=========
terminate called after throwing an instance of 'c10::AcceleratorError'
========= Target application returned an error
========= ERROR SUMMARY: 250 errors
========= ERROR SUMMARY: 248 errors were not printed. Use --print-limit option to adjust
"""


def test_parse():
    report = parse("memcheck", MEMCHECK, 6)
    assert not report.ok and report.errors == 250
    assert report.messages == [
        "Invalid __global__ write of size 2 bytes at linear_relu_kernel (oob.py:46)"
    ]
    clean = parse(
        "racecheck", "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n"
    )
    assert clean.ok and clean.errors == 0
    races = parse(
        "racecheck", "========= RACECHECK SUMMARY: 3 hazards displayed (2 errors, 1 warnings)\n"
    )
    assert not races.ok and races.errors == 2
    crash = parse(
        "memcheck", "Traceback\nImportError: no module\n========= ERROR SUMMARY: 0 errors\n", 1
    )
    assert not crash.ok and crash.errors == 0 and "ImportError" in crash.detail


@pytest.mark.gpu
def test_memcheck_finds_an_unmasked_store(tmp_path):
    kernel = tmp_path / "oob.py"
    code = (PROBLEM / "kernel.py").read_text()
    kernel.write_text(code.replace("mask=mask)", ")"))
    spec = PROBLEM / "spec.yaml"
    try:
        report = sanitize(kernel, "memcheck", spec, "ci")
    except SanitizerError as e:
        pytest.skip(str(e))
    if report.detail and "debugger" in report.detail.lower():
        pytest.skip(report.detail)
    assert report.errors > 0
    assert report.messages[0].startswith("Invalid __global__ write")
    assert "oob.py:46" in report.messages[0]
    assert sanitize(PROBLEM / "kernel.py", "memcheck", spec, "ci").ok
