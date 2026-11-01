# sm-forge

sm-forge makes GPU kernels faster without making them wrong. Give it a correct kernel (Triton or
CUDA C++) and the PyTorch code it implements; it has an LLM rewrite the kernel stage by stage,
and keeps a rewrite only when it still matches PyTorch and is measurably faster on your NVIDIA
GPU. It is modeled on [Xe-Forge](https://github.com/IntelLabs/Xe-Forge), which does the same for
Intel GPUs.

```
seed kernel ──> measure ──> find issues ──> plan stages ──> rewrite, check, time ──> final kernel
                (timing,     (spills, low     (fusion,        (each candidate must     (memcheck and
                 SASS, ncu)   occupancy, ...)  tiling, ...)    match and be faster)     3 A/B wins)
```

## Features

- **Two kernel languages:** Triton, and CUDA C++ with CUTLASS, built in about 5 s per kernel.
- **Checked results:** static checks, correctness against PyTorch
  at odd sizes, a check that no cuBLAS or ATen kernel does the work, compute-sanitizer, and
  interleaved A/B timing with a noise band.
- **Three ways to optimize:** a fixed multi-stage pipeline, a single stage, or a headless Claude
  Code session that works through a trial tree.
- **A CUTLASS tile search** that tunes GEMM configs and compares them with cuBLAS.
- **A knowledge base** of rules and patterns, every one verified by a kernel pair run on the GPU.
- **A corpus** of 138 problems (KernelBench Level 1 and 2, plus inference kernels) with seeds
  and baselines, split into `dev` and a frozen `eval`.

## Tested on

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5080 (sm_120), 16 GB |
| OS | Ubuntu 22.04 under WSL2 on Windows 11 |
| Software | CUDA 12.8, PyTorch 2.11, Triton 3.6, CUTLASS 4.8.1, Python 3.12 |

Other NVIDIA GPUs from sm_80 up should work but are untested. The committed results, the
`hardware/rtx5080.yaml` profile and the corpus shapes all come from the one GPU above.

## Quick start

> [!IMPORTANT]
> You need an NVIDIA GPU with a CUDA 12.8+ driver, Python 3.11+, and for CUDA C++ kernels the
> CUDA toolkit (nvcc), a C++ compiler and git. For the LLM steps you need an Anthropic API key
> or a logged-in [Claude Code](https://claude.com/claude-code).

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

This writes `hardware/<gpu>.yaml`, which the optimizer reads. The profiling tools are optional;
see [profiling tools](docs/profiling-tools.md) to enable them.

### 4. Check and time a kernel (no LLM)

Every problem in `corpus/` has a reference, a spec and a seed kernel. Check one seed and time it
against another:

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

### 5. Optimize a problem

```bash
smforge pipeline --ids l2_057_conv2d_relu_hardswish --max-stages 2 --max-iters 2
```

```
l2_057_conv2d_relu_hardswish: 1.25x [open_ended+ memory_access+ autotuning+] ($0.46)
```

The run analyzed the seed, planned three stages and kept a rewrite from each (`+`; a `-` marks a
stage whose rewrites were all rejected). The final kernel and every candidate are in
`.smforge/runs/<run>/l2_057_conv2d_relu_hardswish/`, and the numbers in `results/pipeline.csv`.
Results vary from run to run. Add `--dsl cuda` to start from the CUDA C++ seed instead.

## What you can run

| To | Run | Guide |
|---|---|---|
| check, time, profile or sanitize one kernel | `smforge-skill check / bench / facts / profile / sanitize` | [harness](docs/harness.md) |
| optimize problems end to end | `smforge pipeline` | [optimization](docs/optimization.md) |
| run one optimization stage | `smforge optimize --stage <stage>` | [optimization](docs/optimization.md) |
| let a Claude Code session optimize | `smforge claude` | [Claude Code engine](docs/claude-code-engine.md) |
| tune CUTLASS GEMM tiles | `smforge tile-search --shapes 4096x4096x4096` | [tile search](docs/tile-search.md) |
| write seed kernels, measure baselines | `smforge seed`, `smforge corpus baselines` | [corpus](docs/corpus.md) |
| inspect or verify the knowledge base | `smforge kb list / show / verify` | [knowledge base](docs/knowledge-base.md) |

Write your own problem by following the [kernel format](docs/kernel-format.md).

## Results on the RTX 5080

| Run | Result | Details |
|---|---|---|
| Triton pipeline, 96 `dev` problems | 80 improved; geomean 2.55x over the seeds; faster than `torch.compile` on 87, geomean 2.36x; $1.07 per problem | [stage 5](PLAN.md#stage-5-analyzer-planner-all-stages) |
| CUDA C++ pipeline, 14 `dev` problems | 13 improved; geomean 2.81x over the CUDA seeds; 0.98x of the Triton finals | [stage 7](PLAN.md#stage-7-cuda-c-and-cutlass-tile-search) |
| CUTLASS tile search, 5 GEMM shapes | within 0.995x to 1.048x of cuBLAS; 1.10x to 1.30x over CUTLASS's default config on 3 shapes | [stage 7](PLAN.md#stage-7-cuda-c-and-cutlass-tile-search) |

## Running tests

```bash
pytest                                  # GPU tests are skipped without an NVIDIA GPU
pytest runners/test_kb_examples.py      # every knowledge base pair on this GPU
ruff check .
```

## Documentation

- [Kernel format](docs/kernel-format.md): reference, kernel and spec files; Triton and CUDA C++.
- [Harness](docs/harness.md): `smforge-skill` commands, what each check does, the trial tree.
- [Optimization](docs/optimization.md): the pipeline and single stages, results, resuming.
- [Claude Code engine](docs/claude-code-engine.md): headless sessions and their workspaces.
- [Tile search](docs/tile-search.md): CUTLASS GEMM configs against cuBLAS.
- [Corpus](docs/corpus.md): the problems, seeds and baselines.
- [Knowledge base](docs/knowledge-base.md): constraints, patterns and how they are verified.
- [Profiling tools](docs/profiling-tools.md): ncu, nsys and compute-sanitizer setup, WSL2 included.
- [Configuration](docs/configuration.md): `.env` settings and where output goes.
- [PLAN.md](PLAN.md): the design, stage by stage, with each stage's measured results.

## Project status

Stages 0 to 7 of [PLAN.md](PLAN.md) are built (v0.7). Stage 8, the report on the frozen `eval`
split with an ablation table, is next.

## License

Apache-2.0; see [LICENSE](LICENSE). The timing loop in `src/smforge/harness/timing.py` is adapted
from [Xe-Forge](https://github.com/IntelLabs/Xe-Forge), also Apache-2.0. The KernelBench problems
in `corpus/kernelbench/` are under KernelBench's MIT license, included there.
