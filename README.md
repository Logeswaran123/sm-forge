# sm-forge

Takes a correct Triton kernel and a shape spec, and returns a faster kernel that passes the same
correctness checks on an NVIDIA GPU. Modeled on [Xe-Forge](https://github.com/IntelLabs/Xe-Forge),
which does the same for Intel GPUs. See [PLAN.md](PLAN.md) for the stages.

## Install

Python 3.11+, a CUDA 12.8+ driver, and torch/triton built for CUDA 12.8:

```bash
uv sync --extra cu128 --extra dev
# or
pip install -e ".[cu128,dev]"
```

To reuse an existing environment that already has torch and triton:

```bash
python -m venv --system-site-packages .venv
.venv/bin/pip install -e ".[dev]"
```

## Kernel format

The Xe-Forge / KernelBench format. A kernel file defines `class Model(nn.Module)` whose
`forward` launches `@triton.jit` kernels. A reference file defines the same `Model` in PyTorch.
A spec gives input shapes, `Model.__init__` arguments and benchmark variants:

```yaml
inputs:
  X: {shape: [M, K], dtype: float16}
inits:
  - in_features: K
  - out_features: N
bench-gpu:
  - params: [X]
    dtype: float16
    dims: {M: 4096, K: 2048, N: 2048}
    flop: "2*M*N*K"
    bytes: "2*(M*K + N*K + M*N)"
```

Tolerances come from the CLI, then the variant (`rtol`, `atol`), then the environment, then a
default for the variant's dtype.

## Commands

```bash
smforge device --all                 # query, probe, measure; writes hardware/<gpu>.yaml

smforge-skill validate kernel.py --spec spec.yaml
smforge-skill check kernel.py --reference reference.py --spec spec.yaml
smforge-skill check kernel.py --baseline seed.py --spec spec.yaml
smforge-skill bench kernel.py --spec spec.yaml
smforge-skill bench kernel.py --baseline seed.py --spec spec.yaml
smforge-skill facts kernel.py --spec spec.yaml
```

Every GPU command runs in a worker subprocess. A kernel that faults the CUDA context kills only
the worker, which is restarted for the next request.

## Corpus

Problems live in `corpus/<source>/<id>/`: `reference.py`, `spec.yaml`, `problem.yaml`, and once
seeded, `kernel.py` with its `seed.yaml`.

```bash
smforge corpus import-kernelbench <KernelBench checkout> --level 2
smforge corpus split                       # dev / eval, stratified by family; eval stays frozen
smforge corpus list --split dev
smforge seed --source inference            # LLM-written seed kernels, verified on the GPU
smforge seed --backend claude-code         # same, through `claude -p` instead of an API key
smforge corpus baselines --modes eager,compile,seed    # writes results/baselines.csv
python scripts/roofline.py                 # plots/roofline.png
```

## Optimization

`smforge optimize` runs one optimization stage on corpus problems, starting from each problem's
seed. The LLM gets the stage's instructions, the knowledge base slice for the problem's family,
the reference, and the current kernel with its timing and per-kernel facts. Each candidate goes
through the static checks, an interface check (`Model.__init__`, `get_inputs`,
`get_init_inputs` unchanged), correctness on `ci` and `bench-gpu`, the launch audit, and an
interleaved A/B against the current kernel. It is kept only if it is faster by more than the
min-effect band; otherwise the input carries forward.

```bash
smforge optimize --stage autotuning --split dev --max-iters 3 --budget-usd 20
smforge optimize --stage autotuning --ids l1_002_standard_matrix_multiplication --record r.jsonl
smforge optimize --stage autotuning --ids l1_002_standard_matrix_multiplication --replay r.jsonl
```

Results go to `results/stages.csv`; candidates, their outcomes and the stage output to
`.smforge/runs/<run>/<problem>/<stage>/`.

## Knowledge base

`knowledge_base/triton/` holds the constraints and optimization patterns the LLM stages see.
Every entry comes with a kernel pair on a dev-split problem. A constraint's `wrong.py` must fail
the way the entry says (static check, error, mismatch, or slower) and its `correct.py` must pass.
A pattern's `after.py` must pass the candidate checks and beat `before.py` in an interleaved
A/B comparison on the GPU. Measured speedups go to `examples/measured.yaml` and stay out of
prompts.

```bash
smforge kb list                              # entries, stages, measured speedups
smforge kb show fusion --families gemm       # a stage's slice as a prompt receives it
smforge kb verify [--ids a,b] [--write]      # run every pair on this GPU
pytest runners/test_kb_examples.py           # the same checks as tests
```

## Tests

```bash
pytest            # GPU tests are skipped when no NVIDIA GPU is present
ruff check .
```

## License

Apache-2.0; see [LICENSE](LICENSE). The timing loop in `src/smforge/harness/timing.py` is adapted
from [Xe-Forge](https://github.com/IntelLabs/Xe-Forge), also Apache-2.0. The KernelBench problems
in `corpus/kernelbench/` are under KernelBench's MIT license, included there.
