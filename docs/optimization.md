# Optimization: one stage, or the whole pipeline

Both commands start from a problem's seed, call the LLM, and keep a candidate only if the
harness says it is correct and faster. Add `--dsl cuda` to work on the CUDA C++ seeds.

## How a candidate is judged

Every candidate goes through these checks in order, and the first failure goes back to the LLM
with the details:

1. Static checks and the audit of `forward` ([kernel format](kernel-format.md)).
2. The interface: `Model.__init__`, `get_inputs` and `get_init_inputs` unchanged.
3. Correctness against the reference on `ci` (odd sizes) and `bench-gpu`, and on `ci` with the
   problem's float32 inputs when the kernel it replaces passes that.
4. The launch audit: no cuBLAS, cuDNN or ATen compute kernel that the current kernel does not
   launch.
5. An interleaved A/B on `bench-gpu` against the current kernel. Only a speedup beyond the A/A
   noise band (2%) counts. Otherwise the current kernel stays.

## One stage: `smforge optimize`

```bash
smforge optimize --stage autotuning --ids l1_002_standard_matrix_multiplication --max-iters 3
smforge optimize --stage memory_access --split dev --budget-usd 20
```

The stages are `algorithmic`, `dtype_fix`, `fusion`, `memory_access`, `tensor_descriptors`,
`persistent_kernel`, `device_specific`, `autotuning` and `open_ended`. The LLM gets the stage's
instructions, the [knowledge base](knowledge-base.md) entries for the problem's family, the
reference, and the current kernel with its timing and per-kernel facts.

Results go to `results/stages.csv`, and candidates and their outcomes to
`.smforge/runs/<run>/<problem>/<stage>/`.

## The whole pipeline: `smforge pipeline`

```bash
smforge pipeline --ids l1_026_gelu --max-stages 2 --max-iters 2
smforge pipeline --split dev --budget-usd 100
smforge pipeline --split dev --from l2_034_convtranspose3d_layernorm_gelu_scaling   # resume
```

For each problem it:

1. Measures the seed: timing, per-kernel facts, the kernels it launches, Nsight Compute counters.
2. Turns the measurements into issues (spills, low occupancy, uncoalesced loads, too many
   launches, ...) and asks the analyzer LLM for issues the numbers cannot show.
3. Plans the stages those issues route to, most valuable first, autotuning last.
4. Runs each stage on the current kernel, measuring again after each accepted one.
5. Finalizes: the last accepted kernel must pass compute-sanitizer memcheck and beat the seed in
   three A/B runs in fresh workers, the slowest counting. If it fails, the stage output before it
   is tried, and the seed is kept when none passes.

The run prints one line per problem:

```
l2_063_gemm_relu_divide: 9.36x [algorithmic+ memory_access+] ($0.59)
```

`+` marks an accepted stage, `-` one whose candidates were all rejected.

Results go to `results/pipeline.csv` (one row per problem: seed and final time, plan, accepted
stages, memcheck, the three A/B speedups and their spread) and `results/pipeline_stages.csv`
(one row per stage). With `--dsl cuda`, to `results/pipeline_cuda.csv` and
`results/pipeline_stages_cuda.csv`.

> [!NOTE]
> If ncu or compute-sanitizer does not work when the run starts, the run continues without it
> (without ncu, the issues that need counters go undetected). If one stops working partway, or
> the LLM backend stops, the run stops and prints the `--from` that resumes it.

## Lessons ledger

Every candidate adds a line to `.smforge/lessons/<gpu>/<problem>.md`: what it changed, and the
check it failed or its speedup. Later runs on the same GPU put the ledger in each stage's prompt.
CUDA runs keep their own ledger.

## Recording and replay

`--record r.jsonl` appends every LLM reply to a file, and `--replay r.jsonl` answers from it without
calling the LLM, to rerun a stage's checks on the same candidates.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--ids`, `--split`, `--source` | all seeded | which problems |
| `--dsl` | `triton` | `cuda` for the CUDA C++ seeds |
| `--max-iters` | 3 (`optimize`), 2 (`pipeline`) | candidates per stage |
| `--budget-usd` | none | stop once the run has spent this much |
| `--backend`, `--claude-bin` | from `.env` | `api` or `claude-code` ([configuration](configuration.md)) |
| `--max-stages` (pipeline) | 3 | stages per problem besides autotuning |
| `--effort` (pipeline) | from `.env` | optimizer effort: low, medium, high, xhigh, max |
| `--from` (pipeline) | none | skip the selected problems before this one |
| `--no-ncu`, `--no-memcheck` (pipeline) | off | run without Nsight Compute, finalize without memcheck |
