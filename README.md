<div align="center">

# sm-forge

### LLM-driven kernel optimization for NVIDIA GPUs

**Faster Triton and CUDA C++ kernels, checked against PyTorch on your own GPU.**

[**Quick start**](#quick-start) · [**Examples**](docs/examples.md) · [**Docs**](#documentation) · [**Results**](#results-on-an-rtx-5080)

</div>

## What is sm-forge?

sm-forge takes a working GPU kernel and tries to make it faster. You give it the matching
PyTorch module, the input shapes you care about, and the kernel code in Triton or CUDA C++. An
LLM proposes changes, and sm-forge builds and tests each version. It checks that the output
matches PyTorch's, that the kernel doesn't pass the work off to cuBLAS or another library, and
that it runs faster than the current version. Anything incorrect or without a clear speedup gets
discarded. The project started as an NVIDIA counterpart to Intel's
[Xe-Forge](https://github.com/IntelLabs/Xe-Forge).

There are three ways to run it:

|  | You give it | It does | Use it for |
|---|---|---|---|
| **Pipeline** | one or more problems (see below) | measures each kernel, lists what slows it down, and has the LLM fix those one at a time | optimizing many problems in one run |
| **Claude Code engine** | the same | a Claude Code session decides what to try, and sm-forge checks and times each attempt and keeps the fastest | spending longer on one problem |
| **Tile search** | matrix multiply sizes | has the LLM pick CUTLASS tile settings, and times them against cuBLAS | tuning a matmul for one size |

The repository comes with 138 problems ready to optimize, a knowledge base of optimization rules
that the LLM is shown, and the results from an RTX 5080.

### Terms used in these docs

| Term | Meaning |
|---|---|
| problem | one thing to optimize: a folder with a PyTorch reference, a spec and a starting kernel. The ones in `corpus/` come from [KernelBench](https://github.com/ScalingIntelligence/KernelBench) and common inference kernels |
| reference | `reference.py`, the PyTorch module whose output every kernel must match |
| spec | `spec.yaml`, the input shapes and dtypes. `ci` lists small, odd sizes used only to check correctness, and `bench-gpu` the size that is timed |
| seed | the starting kernel (`kernel.py` for Triton, `kernel_cuda.py` for CUDA C++). It is correct but not tuned, and speedups are measured against it |
| `dev` and `eval` | the two groups of problems. Development used `dev` only, and `eval` was kept back to check the results hold on problems the tool was not tuned on |
| A/B run | the two kernels timed in alternating rounds, so a slowdown of the machine hits both. A speedup under 2% counts as no change |

## Tested on

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5080 (sm_120), 16 GB |
| OS | Ubuntu 22.04 under WSL2 on Windows 11 |
| Software | CUDA 12.8, PyTorch 2.11, Triton 3.6, CUTLASS 4.8.1, Python 3.12 |

Other NVIDIA GPUs from sm_80 up should work but are untested. The committed results, the
`hardware/rtx5080.yaml` profile and the problem shapes all come from the one GPU above.

## Quick start

> [!IMPORTANT]
> You need an NVIDIA GPU with a CUDA 12.8+ driver, Python 3.11+, and for CUDA C++ kernels the
> CUDA toolkit (nvcc), a C++ compiler and git. For the LLM steps you need an Anthropic API key
> or a logged-in [Claude Code](https://claude.com/claude-code).

Steps 1 to 4 make no LLM calls. Step 5 does, and an optimized problem costs about $1 in LLM
calls.

### 1. Install

```bash
git clone https://github.com/Logeswaran123/sm-forge.git
cd sm-forge
uv sync --extra cu128 --extra dev     # PyTorch and Triton for CUDA 12.8, ninja, pytest, ruff
source .venv/bin/activate
```

Without [uv](https://docs.astral.sh/uv/):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[cu128,dev]" --extra-index-url https://download.pytorch.org/whl/cu128
```

> [!TIP]
> Already have PyTorch and Triton installed? Use them instead:
> `python -m venv --system-site-packages .venv && .venv/bin/pip install -e ".[dev]" ninja`

### 2. Choose how to reach the LLM

```bash
cp .env.example .env
```

Then either put your key in `.env` (`ANTHROPIC_API_KEY=...`), or set `LLM_BACKEND=claude-code`
to use Claude Code's login instead. See [configuration](docs/configuration.md).

### 3. Profile your GPU

```bash
smforge device --all      # probes features, measures bandwidth, TFLOPS and timing noise
smforge device --tools    # which of ncu, nsys and compute-sanitizer work here
```

This takes a few minutes and writes `hardware/<gpu>.yaml`, which the optimizer reads. The
profiling tools are optional. See [profiling tools](docs/profiling-tools.md) to enable them.

### 4. Check and time a kernel (no LLM)

There are two commands. `smforge-skill` works on one kernel file and never calls the LLM.
`smforge` runs the optimizer and the other tools. Take a matrix multiply problem that has both a
Triton and a CUDA C++ seed, check the Triton one against PyTorch, then time it against the CUDA
one:

```bash
P=corpus/kernelbench/l1_002_standard_matrix_multiplication
smforge-skill check $P/kernel.py --reference $P/reference.py --spec $P/spec.yaml
smforge-skill bench $P/kernel.py --baseline $P/kernel_cuda.py --spec $P/spec.yaml
```

```
correctness: pass (2 draws, rtol 0.01, atol 0.01, weights none)
...
baseline 10913.8 us, candidate 1165.3 us (p20): 9.366x, faster (min effect 2.0%)
```

`check` passed: on two sets of random inputs the kernel's output matched PyTorch's within the
tolerances. In `bench`, the first file is the candidate and `--baseline` the one it is compared
with, so the Triton seed is 9.4x faster than the CUDA seed. The CUDA seed is a simple kernel
without tensor cores, while the Triton one uses them. `p20` means each time is the 20th
percentile of many runs.

### 5. Optimize a problem

```bash
smforge pipeline --ids l2_057_conv2d_relu_hardswish --max-stages 2 --max-iters 2
```

```
l2_057_conv2d_relu_hardswish: 1.25x [open_ended+ memory_access+ autotuning+] ($0.46)
```

This takes a few minutes. The line gives the speedup over the seed, the optimization stages
that ran (`+` if the LLM's change was kept, `-` if every attempt was rejected), and what the LLM
calls cost. Results vary from run to run.

The final kernel and every attempt are in `.smforge/runs/<run>/`. The run also replaces this
problem's row in `results/pipeline.csv`, which holds the published numbers, so
`git checkout results/` puts them back. The [examples](docs/examples.md) walk through reading a
run like this one, and through everything else below.

## What you can run

| To | Run | Guide |
|---|---|---|
| check, time, profile or sanitize one kernel | `smforge-skill check / bench / facts / profile / sanitize` | [harness](docs/harness.md) |
| optimize problems end to end | `smforge pipeline` | [optimization](docs/optimization.md) |
| apply one kind of rewrite (fusion, tiling, ...) | `smforge optimize --stage fusion` | [optimization](docs/optimization.md) |
| let a Claude Code session optimize | `smforge claude` | [Claude Code engine](docs/claude-code-engine.md) |
| tune CUTLASS GEMM tiles | `smforge tile-search --shapes 4096x4096x4096` | [tile search](docs/tile-search.md) |
| write starting kernels, measure baselines | `smforge seed`, `smforge corpus baselines` | [corpus](docs/corpus.md) |
| inspect or verify the knowledge base | `smforge kb list / show / verify` | [knowledge base](docs/knowledge-base.md) |
| evaluate a split, compare configs | `smforge eval run / report` | [evaluation](docs/evaluation.md) |

To work on your own kernel, see [bring your own kernel](docs/examples.md#6-bring-your-own-kernel).

## Results on an RTX 5080

Speedups are geometric means over the problems, with Opus 5.5 as the LLM. `torch.compile` and
max-autotune are PyTorch's own compiler at its default and most thorough settings.

| Run | Result | Table |
|---|---|---|
| Triton pipeline, 96 `dev` problems | 80 faster than their seed, 2.55x on average. Faster than `torch.compile` on 87, by 2.36x on average. $1.07 per problem | `results/pipeline.csv` |
| Triton pipeline, 42 `eval` problems | 36 faster than their seed and none slower, 2.73x on average. 2.61x faster than `torch.compile` and 2.27x faster than max-autotune. $1.03 per problem | `results/eval/eval/` |
| CUDA C++ pipeline, 14 `dev` problems | 13 faster than their seed, 2.81x on average, ending about level with the Triton results (0.98x) | `results/pipeline_cuda.csv` |
| CUTLASS tile search, 5 matmul sizes | Between 0.5% slower and 4.8% faster than cuBLAS. 1.10x to 1.30x faster than CUTLASS's default settings on 3 sizes | `results/tilesearch.csv` |
| The same 12 `dev` problems under 13 settings | The choice of model matters most: 1.71x with Haiku 5.5, 2.04x with Opus 5.5, 2.13x with Opus 5.5 at high effort. Turning off the knowledge base or other parts of the pipeline changed the result by 3% or less, with one outlier problem left out | `results/eval/ablations.md` |

## Running tests

```bash
pytest                                  # GPU tests are skipped without an NVIDIA GPU
pytest runners/test_kb_examples.py      # every knowledge base pair on this GPU
ruff check .
```

## Documentation

New to the project? Start with the examples, then the kernel format.

- [Examples](docs/examples.md): walkthroughs, from timing a kernel to bringing your own.
- [Kernel format](docs/kernel-format.md): reference, kernel and spec files, in Triton and CUDA C++.
- [Harness](docs/harness.md): `smforge-skill` commands, what each check does, the trial tree.
- [Optimization](docs/optimization.md): the pipeline and single rewrites, results, resuming.
- [Claude Code engine](docs/claude-code-engine.md): headless sessions and their workspaces.
- [Tile search](docs/tile-search.md): CUTLASS GEMM configs against cuBLAS.
- [Corpus](docs/corpus.md): the problems, starting kernels and baselines.
- [Knowledge base](docs/knowledge-base.md): constraints, patterns and how they are verified.
- [Evaluation](docs/evaluation.md): eval runs, held-out checks and ablations.
- [Profiling tools](docs/profiling-tools.md): ncu, nsys and compute-sanitizer setup, WSL2 included.
- [Configuration](docs/configuration.md): `.env` settings and where output goes.
- [PLAN.md](PLAN.md): design notes and measured results.

## Citation & license

```bibtex
@software{smforge2026,
  title  = {sm-forge: LLM-driven kernel optimization for NVIDIA GPUs},
  author = {Sivakumar, Logeswaran},
  year   = {2026},
  url    = {https://github.com/Logeswaran123/sm-forge}
}
```

Source code is released under [Apache-2.0](LICENSE). The timing loop in
`src/smforge/harness/timing.py` is adapted from [Xe-Forge](https://github.com/IntelLabs/Xe-Forge)
(Apache-2.0), and the KernelBench problems in `corpus/kernelbench/` are under KernelBench's MIT
license. This project downloads third-party open source software (PyTorch, Triton, CUTLASS).
Review those licenses before use.
