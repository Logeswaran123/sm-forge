# Evaluation

`smforge eval run` runs the pipeline over a split with one fixed config, and measures each result
past what the pipeline itself checked. `smforge eval report` puts runs side by side.

```bash
smforge eval run --name eval --split eval
smforge eval report --runs eval
```

```bash
# ablations: the same problems under different configs, the first run as the baseline
IDS=l1_035_groupnorm,l2_057_conv2d_relu_hardswish
smforge eval run --name base --ids $IDS
smforge eval run --name no-kb --ids $IDS --no-kb
smforge eval run --name t5 --ids $IDS --max-iters 5
smforge eval report --runs base,no-kb,t5 --output results/eval/ablations.md
```

A run lives in `results/eval/<name>/`. Rerunning the same command resumes it, skipping the
problems already done. A run's config is fixed: running a name again with other options stops
with an error, so use a new name. Eval runs neither read nor write the lessons ledger, so earlier
runs on the same problems cannot leak into the results.

## What is measured

| Measure | How |
|---|---|
| speedup over the seed | the slowest of the final A/B runs against the seed, as in the pipeline |
| kernel-only speedup | the summed kernel time of one forward, seed against final, from the profiler |
| speedup over eager, `torch.compile`, max-autotune | the final time against the times in `results/baselines.csv` |
| held-out correctness | `ci` and `bench-gpu` on input draws 1000 and up (optimization uses draws 0 and 1), and the problem's original KernelBench shapes at the timed dtype, which no check during optimization saw |
| roofline fraction | the time the measured matmul peak and DRAM bandwidth allow for the spec's FLOPs and bytes, over the final time. Above 1 means the kernel does less work than the reference's FLOP count. |
| LLM use | calls, input, output and cached tokens, the cache-hit share, dollars at list price |
| wall time | per problem, all pipeline runs and measurements included |

The run's summary counts a problem as improved when its final is more than 2% faster than its
seed, and as a regression when it is more than 2% slower.

Max-autotune baselines are not measured by default. Add them before an eval run with
`smforge corpus baselines --modes max-autotune --split eval`.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--name` | required | the run's directory under `results/eval/` |
| `--split`, `--ids` | `eval` | which problems |
| `--dsl` | `triton` | `cuda` for the CUDA C++ seeds |
| `--max-iters` | 2 | candidates per stage (T) |
| `--max-stages` | 2 | stages per problem besides autotuning |
| `--best-of` | 1 | whole pipeline runs per problem, the fastest kept |
| `--model`, `--effort` | from `.env`, `medium` | the model for every role, and the optimizer's effort |
| `--no-kb` | off | no knowledge base in the prompts |
| `--no-arch-gating` | off | knowledge base entries for every GPU, not only this one |
| `--no-facts` | off | prompts show the timing only: no per-kernel facts, launches or counters, no fact-backed issues, no fact deltas in feedback |
| `--no-reanalysis` | off | no measurement after an accepted stage, so later stages work from the seed's |
| `--no-rollback` | off | a stage keeps its fastest correct candidate even if slower, and the last stage's output is final |
| `--no-ncu`, `--no-memcheck` | off | as in the [pipeline](optimization.md#options) |

## Output

```
results/eval/<name>/
  config.json          # the run's config
  device.json          # GPU, driver, library versions, git commit
  <problem>.json       # one problem's measurements
  summary.json         # the run's aggregates
```

Every pipeline run's candidates are under `.smforge/runs/<stamp>/<name>/<problem>/`.
