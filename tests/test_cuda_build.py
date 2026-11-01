from pathlib import Path

import pytest

from smforge.cuda import BuildError, build, loaded, parse_log

FIXTURE = Path(__file__).parent / "kernels" / "linear_relu" / "kernel_cuda.py"

NVCC_LOG = """\
Error building extension 'smf_1': [1/3] /usr/local/cuda/bin/nvcc ... -c /c/smf_1/cuda.cu -o cuda.cuda.o
FAILED: [code=2] cuda.cuda.o
/c/smf_1/cuda.cu(3): error: identifier "acc" is undefined
    acc += 1;
    ^
/c/cutlass/include/cutlass/gemm/gemm.h(120): error: static assertion failed with "bad tile"
          detected during:
            instantiation of "Gemm<Tile>" at line 4 of /c/smf_1/cuda.cu
2 errors detected in the compilation of "/c/smf_1/cuda.cu".
/c/smf_1/main.cpp:2:10: error: 'tensor' was not declared in this scope
ptxas error   : Entry function '_Z1kv' uses too much shared data (0x1c000 bytes, 0x18c00 max)
ninja: build stopped: subcommand failed.
"""


def test_parse_log():
    cuda = "#include <cuda_runtime.h>\n__global__ void k() {\n  acc += 1;\n  Gemm<Tile> g;\n}"
    cpp = "void f();\nauto t = tensor();"
    kernel_file = f'X = 1\nCUDA_SRC = r"""{cuda}"""\nCPP_SRC = r"""{cpp}"""\n'
    found = parse_log(NVCC_LOG, cuda, cpp, kernel_file)
    assert [str(d) for d in found] == [
        'line 4: identifier "acc" is undefined',
        'line 5: static assertion failed with "bad tile" (in include/cutlass/gemm/gemm.h:120)',
        "line 8: 'tensor' was not declared in this scope",
        "cuda source: ptxas: Entry function '_Z1kv' uses too much shared data "
        "(0x1c000 bytes, 0x18c00 max)",
    ]
    assert parse_log(NVCC_LOG, cuda, cpp)[0].line is None


def test_torch_header_in_cuda_source():
    with pytest.raises(BuildError, match="includes a PyTorch header"):
        build("#include <cuda_runtime.h>\n#include <torch/extension.h>\n", "", ["f"])


@pytest.mark.gpu
def test_build(tmp_path):
    import torch

    from smforge.harness.workload import load_module

    module = load_module(FIXTURE)
    assert any(path.exists() for path in loaded().values())
    torch.manual_seed(0)
    model = module.Model(96, 200).cuda()
    for dtype in (torch.float16, torch.float32):
        x = torch.randn(127, 96, device="cuda", dtype=dtype)
        expected = torch.relu(x @ model.weight.to(dtype).T + model.bias.to(dtype))
        got = module.Model.forward(model.to(dtype), x)
        torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)

    broken = tmp_path / "broken.py"
    text = FIXTURE.read_text()
    broken.write_text(text.replace("float acc = 0.0f;", "float acc = undefined_name;"))
    line = text.splitlines().index("  float acc = 0.0f;") + 1
    with pytest.raises(BuildError, match=f'line {line}: identifier "undefined_name" is undefined'):
        load_module(broken)

    missing = tmp_path / "missing.py"
    missing.write_text(text.replace("void linear_relu(int dtype", "void other(int dtype", 1))
    with pytest.raises(BuildError, match="undefined symbol linear_relu"):
        load_module(missing)
