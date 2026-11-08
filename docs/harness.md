# The harness: checking and timing kernels

`smforge-skill` runs a kernel file through the same checks and timing the optimizer uses. It
needs no LLM. Every GPU command runs in a worker subprocess, so a kernel that faults the CUDA
context kills only the worker, which restarts for the next request.

## Commands

```bash
smforge-skill validate kernel.py --spec spec.yaml                      # static checks, no GPU
smforge-skill check kernel.py --reference reference.py --spec spec.yaml
smforge-skill check kernel.py --baseline seed.py --spec spec.yaml      # another kernel as oracle
smforge-skill bench kernel.py --spec spec.yaml
smforge-skill bench kernel.py --baseline seed.py --spec spec.yaml      # interleaved A/B
smforge-skill facts kernel.py --spec spec.yaml
smforge-skill sanitize kernel.py --spec spec.yaml --variant ci --tool memcheck
smforge-skill profile kernel.py --spec spec.yaml --baseline seed.py
```

`--variant` picks a spec variant (`ci`, `bench-gpu`, ...). Each command has its own default.

| Command | What it does |
|---|---|
| `validate` | Static checks and the audit of `forward`: no PyTorch compute, no state kept on `self`, no host syncs, and the language's own rules (block shapes for Triton, launch checks for CUDA). |
| `check` | Runs the kernel and the oracle on two input draws and compares the outputs, then lists the kernels one forward launches and rejects library or ATen compute kernels. Autotuned Triton kernels are checked with every config. |
| `bench` | Device time at the spec's shape, a quantile of many samples with the L2 cache flushed. With `--baseline`, an interleaved A/B: the verdict is faster or slower only outside the A/A noise band, how much a kernel's time moves when timed against itself (`BENCHMARK_MIN_EFFECT`, 2%). |
| `facts` | Per kernel: grid, block, registers, spills, shared memory, resident CTAs per SM, waves, and the SASS instruction mix. |
| `sanitize` | compute-sanitizer (`memcheck`, `racecheck`, `initcheck` or `synccheck`) on a warm-up forward and one more, with PyTorch's caching allocator off so an out-of-bounds access does not land in a cached block. |
| `profile` | Nsight Compute's speed-of-light, memory, occupancy, launch, warp-state and source sections on the first launch of each kernel, about 30 lines per kernel, and with `--baseline` what changed. |

Example, the Triton seed against the CUDA seed of one corpus problem:

```
$ P=corpus/kernelbench/l1_002_standard_matrix_multiplication
$ smforge-skill bench $P/kernel.py --baseline $P/kernel_cuda.py --spec $P/spec.yaml
baseline 10913.8 us, candidate 1165.3 us (p20): 9.366x, faster (min effect 2.0%)
rounds: 9.372 9.365 9.179 9.987 10.013 10.023 9.192 9.360 9.364 9.366
```

`sanitize` and `profile` need compute-sanitizer and ncu. See [profiling tools](profiling-tools.md).

## Trial tree

A trial tree keeps every candidate for one problem, with its parent and what the harness
measured. `t0` is the baseline. `save` stores a kernel and checks it against `t0` at once, A/B
timing included, so no result can be entered by hand. `finalize` copies the fastest trial that
beats `t0` and passes the final check (memcheck, then three A/B runs against `t0` in fresh
workers, the slowest counting). Parity and regressions are never finalized.

```bash
smforge-skill trial init --problem l1_002_standard_matrix_multiplication --dir trials
smforge-skill trial save work/k.py --parent t0 --strategy "128x128 tiles, 4 stages" --dir trials
smforge-skill trial status --dir trials
smforge-skill trial best --dir trials
smforge-skill trial finalize output.py --dir trials
```

The [Claude Code engine](claude-code-engine.md) works through this tree.

## Timing on a desktop GPU

A forward that launches many short kernels is timed with the host's launch speed, which on a
desktop (and under WSL2) drops for seconds at a time. One A/B can land in a fast window and
overstate a speedup. The pipeline therefore finalizes a kernel only if it wins three A/B runs in
fresh workers, and counts the slowest.
