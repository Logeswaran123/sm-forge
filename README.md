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

## Setup

```bash
cp .env.example .env        # set ANTHROPIC_API_KEY, or use --backend claude-code
smforge device --all        # once per GPU: writes hardware/<gpu>.yaml
smforge device --tools      # which of ncu, nsys and compute-sanitizer work here
```

`smforge seed`, `optimize` and `pipeline` call the Anthropic API with `ANTHROPIC_API_KEY`.
`--backend claude-code` runs headless Claude Code on its own login instead, using `claude` on
`PATH` or `CLAUDE_BIN`. The harness commands (`smforge-skill`, `smforge kb verify`,
`smforge corpus baselines`) need no LLM.

`smforge device --all` probes the GPU's features and measures its roofline and timing noise.
Without a profile for the GPU, the knowledge base is not filtered for the device and the
analyzer has no SM count or roofline. The committed `hardware/rtx5080.yaml`, the numbers in
`results/` and the corpus `bench-gpu` shapes come from the one GPU sm-forge has been developed
on, an RTX 5080 under WSL2.

ncu, nsys and compute-sanitizer come with the CUDA toolkit and are found on `PATH`, under
`CUDA_HOME`, or in `/usr/local/cuda*`. They are optional; without ncu the pipeline skips the
issues that need counters (`uncoalesced_access`, `bank_conflict`).

- ncu needs access to the GPU performance counters
  ([ERR_NVGPUCTRPERM](https://developer.nvidia.com/ERR_NVGPUCTRPERM)). On Linux, load the
  driver with `NVreg_RestrictProfilingToAdminUsers=0`. Under WSL2 the setting is on Windows:
  NVIDIA Control Panel, Desktop > Enable Developer Settings, then Developer > Manage GPU
  Performance Counters > Allow access to all users; then `wsl --shutdown`.
- compute-sanitizer under WSL2 needs the Windows debugger interface: run
  `EnableDebuggerInterface.bat` from the Windows CUDA toolkit as administrator, then reboot.

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
smforge-skill sanitize kernel.py --spec spec.yaml --variant ci --tool memcheck
smforge-skill profile kernel.py --spec spec.yaml --baseline seed.py
```

`profile` runs Nsight Compute's SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, LaunchStats,
WarpStateStats and SourceCounters sections on the first launch of each Triton kernel in one
forward, prints about 30 lines per kernel with ncu's whole-kernel estimates, and with
`--baseline` the metrics that changed.

`sanitize` runs compute-sanitizer (`memcheck`, `racecheck`, `initcheck` or `synccheck`) on a
warm-up forward, which runs every autotune config, and one more forward, with PyTorch's caching
allocator off so that an out-of-bounds access does not land in a cached block.

Every GPU command runs in a worker subprocess. A kernel that faults the CUDA context kills only
the worker, which is restarted for the next request.

A trial tree keeps every candidate for one problem, with its parent and what the harness
measured. `t0` is the baseline. `save` stores a kernel and checks it against `t0` at once (the
checks `smforge optimize` runs, A/B timing included), so no result can be entered by hand.
`finalize` copies the fastest trial that beats `t0` and passes memcheck; parity and regressions
are never finalized.

```bash
smforge-skill trial init --problem l1_002_standard_matrix_multiplication --dir trials
smforge-skill trial save work/k.py --parent t0 --strategy "128x128 tiles, 4 stages" --dir trials
smforge-skill trial status --dir trials
smforge-skill trial finalize output.py --dir trials
```

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

`smforge pipeline` runs the whole optimization on a problem. It measures the seed (timing, SASS
facts, launches, Nsight Compute counters), turns the measurements into fact-backed issues, asks
the analyzer for the issues the numbers cannot show, and plans the stages those issues route to.
Each stage gets its issues; after an accepted stage the kernel is measured again. The final
kernel must pass compute-sanitizer memcheck on `ci` (one that fails gives way to the stage output
before it) and is kept only if it beats the seed.

```bash
smforge pipeline --split dev --max-iters 3 --budget-usd 100
smforge pipeline --split dev --from l2_034_convtranspose3d_layernorm_gelu_scaling
```

If ncu or compute-sanitizer does not work when the run starts, the run goes on without it. If
one stops working partway (counter access withdrawn, say), the run stops and prints the `--from`
that resumes it, as it does when the LLM backend stops.

Results go to `results/pipeline.csv` (one row per problem) and `results/pipeline_stages.csv`
(one row per problem and stage).

Every candidate and every trial also adds a line to the problem's lessons ledger,
`.smforge/lessons/<gpu>/<problem>.md`: what it changed, the check it failed or its speedup. Later
pipeline runs on the same GPU put the ledger in each stage's prompt.

## Claude Code engine

`smforge claude` gives each problem to a headless Claude Code session in a workspace of its own
(`.smforge/workspaces/<run>/<problem>/`): the seed as `kernel.py` and trial `t0`, read-only
copies of the reference and spec, a `CLAUDE.md` with the rules and the measurements of `t0`, the
knowledge base filtered for the GPU, the stage instructions, the problem's lessons ledger, and an
`/optimize-kernel` command. The session may write only under `work/` and run only
`smforge-skill`; it saves each candidate with `smforge-skill trial save`, which checks and times
it against `t0`. When it stops (or reaches `--max-turns`), sm-forge finalizes the trial tree
itself, memcheck included, and checks the winner again in its own worker against the corpus
reference.

```bash
smforge claude --ids l1_035_groupnorm --max-trials 8 --max-turns 80 --budget-usd 10
```

It uses `claude` on `PATH` or `CLAUDE_BIN`, logged in. Results go to `results/claude.csv`; the
session's transcript stays in the workspace as `transcript.jsonl`.

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
