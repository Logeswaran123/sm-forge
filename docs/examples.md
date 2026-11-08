# Examples

Walkthroughs on problems from `corpus/`. Run them from the repository root after the
[quick start](../README.md#quick-start). The outputs below are from an RTX 5080. Your times will
differ, and LLM runs differ from one run to the next.

- [1. Check and time a kernel](#1-check-and-time-a-kernel)
- [2. Look at what a kernel compiles to](#2-look-at-what-a-kernel-compiles-to)
- [3. Optimize a kernel and read what happened](#3-optimize-a-kernel-and-read-what-happened)
- [4. Let a Claude Code session optimize it](#4-let-a-claude-code-session-optimize-it)
- [5. Tune a CUTLASS GEMM for a shape](#5-tune-a-cutlass-gemm-for-a-shape)
- [6. Bring your own kernel](#6-bring-your-own-kernel)
- [7. See what the LLM is told](#7-see-what-the-llm-is-told)

## 1. Check and time a kernel

No LLM needed. `l1_002_standard_matrix_multiplication` is an fp16 GEMM with a Triton seed and a
CUDA C++ seed.

```
$ P=corpus/kernelbench/l1_002_standard_matrix_multiplication

$ smforge-skill validate $P/kernel_cuda.py --spec $P/spec.yaml
no issues

$ smforge-skill check $P/kernel_cuda.py --reference $P/reference.py --spec $P/spec.yaml
correctness: pass (2 draws, rtol 0.01, atol 0.01, weights none)
launches per forward, 10546.8 us of kernel time:
  cuda            1x    10546.8 us  void gemm_kernel<__half>(__half const*, __half const*, __half*, int, int, int)
launch audit: no issues
```

`validate` is source-only. `check` runs the kernel next to the PyTorch reference, compares the
outputs, and lists every kernel one forward launched, so a cuBLAS call hidden in the kernel file
would show up here and fail.

Time the Triton seed against the CUDA one:

```
$ smforge-skill bench $P/kernel.py --baseline $P/kernel_cuda.py --spec $P/spec.yaml
baseline 10913.8 us, candidate 1165.3 us (p20): 9.366x, faster (min effect 2.0%)
rounds: 9.372 9.365 9.179 9.987 10.013 10.023 9.192 9.360 9.364 9.366
```

The first file is the candidate and `--baseline` the kernel it is compared with. The two run in
alternating rounds, ten of them, and the verdict says faster only beyond the 2% noise band. The
CUDA seed is a simple kernel without tensor cores, while the Triton seed uses them through
`tl.dot`.

## 2. Look at what a kernel compiles to

`l2_057_conv2d_relu_hardswish` is a 3x3 convolution followed by ReLU and HardSwish.

```
$ P=corpus/kernelbench/l2_057_conv2d_relu_hardswish
$ smforge-skill facts $P/kernel_cuda.py --spec $P/spec.yaml
conv_relu_hswish_kernel<__half>: grid 254016x1x1 (254016 CTAs), block 256x1x1 (8 warps)
  regs 40, spills 0, smem 0.0 KB, 6 CTAs/SM, 504.00 waves
  sass: 608 instructions; FFMA 29, FADD 3, FMUL 1, HFMA2 14, LDG 31, STG 1
  FFMA share of fp32 FFMA/FADD/FMUL: 88%
arithmetic intensity 63.8 FLOP/byte; ridge point 131.2, so memory-bound
```

A CTA is a thread block, and a wave is one round of blocks filling every SM. So this is one
thread per output, no shared memory, no tensor-core instructions (`HMMA`), and 504 rounds of
blocks. The arithmetic intensity says it could be memory-bound, but this kernel is far from
either limit. These are the numbers the optimizer starts from.

## 3. Optimize a kernel and read what happened

```
$ smforge pipeline --dsl cuda --ids l2_057_conv2d_relu_hardswish --max-stages 2 --max-iters 2
WARNING smforge.agents.pipeline: l2_057_conv2d_relu_hardswish: the dtype_fix output's A/B runs spread 7.9%, beyond the A/A band; the slowest counts
l2_057_conv2d_relu_hardswish: 30.29x [algorithmic+ dtype_fix+] ($1.03)
```

Two stages were planned, `algorithmic` and `dtype_fix`, and the LLM's change was kept in both
(`+`). The final kernel beat the seed in three A/B runs in fresh processes (30.29x, 30.29x,
32.69x). They differed by more than the A/A band, the noise seen when a kernel is timed against
itself, so the slowest counts. The warning says the same.

Everything the run did is under `.smforge/runs/<run>/l2_057_conv2d_relu_hardswish/pipeline/`:

```
pipeline.json            # analysis, issues, plan, per-stage results, final times, cost
output.py                # the final kernel
algorithmic/
  attempt-0.py           # each candidate the LLM wrote
  attempts.json          # its outcome: accepted, or which check failed and why
  output.py              # the kernel this stage handed on
dtype_fix/
  ...
```

The analyzer's summary in `pipeline.json` is a good first read:

> Nearly all the time (5874 us) is in one direct-convolution kernel where each thread computes
> one output with 64-bit div/mod indexing, 72 scalar fp16 loads converted to fp32, and 72 fp32
> weight loads from global. Nothing is reused across threads, so the kernel is limited by
> instruction issue (SM 82%, DRAM 3%) and reaches only 1.4 TFLOPS. The work is a small implicit
> GEMM (M≈1.02M, N=64, K=72) whose real floor is writing 130 MB of fp16 output, about 100 us.

The issues it found, the first from the ncu counters and the rest from the analyzer reading the
code: `uncoalesced_access` (global loads use 17.2 of 32 bytes per sector), `no_tensor_core_mma`,
`upcast_operands`, `per_call_weight_prep`, and an open-ended note that no data is reused across
threads. Each issue maps to an [optimization stage](optimization.md#stages), and these mapped to
two. In `attempts.json`, the first stage's accepted candidate says what it changed:

> I replaced the scalar direct convolution with an implicit GEMM on tensor cores for fp16 and
> bf16 inputs. It uses `mma.sync` m16n8k16 with fp32 accumulation...

Check the result yourself:

```
$ F=.smforge/runs/<run>/l2_057_conv2d_relu_hardswish/pipeline/output.py
$ smforge-skill bench $F --baseline $P/kernel_cuda.py --spec $P/spec.yaml
baseline 6419.6 us, candidate 204.7 us (p20): 31.365x, faster (min effect 2.0%)

$ smforge-skill facts $F --spec $P/spec.yaml
conv_mma_kernel<__half, false>: grid 1x21x64 (1344 CTAs), block 256x1x1 (8 warps)
  regs 100, spills 0, smem 44.3 KB, 2 CTAs/SM, 8.00 waves
  sass: 2504 instructions; HMMA 120, FFMA 172, FADD 66, FMUL 32, LDG 32, STG 8, LDS 23, STS 27, LDSM 60, BAR 3
```

Tensor-core instructions, shared-memory tiles loaded with `ldmatrix` (`LDSM`), and 8 waves
instead of 504. At 205 us it is close to the analyzer's 100 us estimate for writing the output.

Without `--dsl cuda` the same command starts from the Triton seed. The numbers go to
`results/pipeline.csv` (`results/pipeline_cuda.csv` here). More in [optimization](optimization.md).

## 4. Let a Claude Code session optimize it

Instead of fixed stages, a headless Claude Code session can decide what to try. sm-forge checks
and times every attempt, then picks the winner itself.

```
$ smforge claude --ids l2_057_conv2d_relu_hardswish --max-trials 8 --max-turns 80
l2_057_conv2d_relu_hardswish: 1.29x (t1; 3 trials, 18 turns, success, $0.59); trial 1.294x, re-measured 1.294/1.294/1.294x
  workspace: .smforge/workspaces/<run>/l2_057_conv2d_relu_hardswish
```

This started from the Triton seed. The session saved three trials, each with its own strategy
note:

```
t1  1.294x  transpose to acc[oc,pixel] per-image tiles with constexpr shapes for vectorized stores, full OC per program, autotune
t2  1.279x  evict_first output stores, wider autotune space (stages 1, 2-warp 64 tiles)
t3  1.100x  persistent grid of CTAS_PER_SM*84 programs looping over pixel tiles, bias hoisted
```

`t1` won, passed memcheck, and held at 1.294x in three more A/B runs. The workspace keeps the
session's transcript and every trial. More in [Claude Code engine](claude-code-engine.md).

## 5. Tune a CUTLASS GEMM for a shape

```
$ smforge tile-search --shapes 2048x4096x8192:row --rounds 3 --per-round 6
2048x4096x8192_float16_row: cuBLAS 1217.5 us; default 128x256x64_w64x64x64_s3: invalid, - (needs 144 KB of shared memory; a block gets 99 KB); fitted 128x256x64_w64x64x64_s2: ok, 1481.8 us
  best 64x64x64_w32x32x64_s4_sw4: A/B 1186.8 us (115.8 TFLOPS) against default 1483.8 us (1.250x) and cuBLAS 1216.5 us (1.024x)
```

CUTLASS's default config does not fit this GPU's shared memory, so the baseline is the same tile
with fewer pipeline stages (shared-memory buffers). After 18 configs the best runs 1.25x faster
than that and 2.4% faster than cuBLAS. Every config tried is in
`.smforge/runs/<run>/tilesearch/<shape>/trials.json`, and each one is a kernel file you can run with `smforge-skill`. More in [tile search](tile-search.md).

## 6. Bring your own kernel

Put a problem in `corpus/custom/<id>/`:

```
corpus/custom/my_linear_relu/
  reference.py     # class Model(nn.Module) in plain PyTorch
  spec.yaml        # shapes, dtypes, the ci and bench-gpu variants
  problem.yaml     # three lines, below
  kernel.py        # your kernel, or let `smforge seed --ids my_linear_relu` write one
```

```yaml
id: my_linear_relu
source: custom
family: gemm
```

`tests/kernels/linear_relu/` has a complete reference, spec and kernel to copy, and
[kernel format](kernel-format.md) explains each file. The problem then works like any other:

```
$ smforge corpus list | grep my_linear
my_linear_relu                                              gemm             -     seeded
$ D=corpus/custom/my_linear_relu
$ smforge-skill check $D/kernel.py --reference $D/reference.py --spec $D/spec.yaml
correctness: pass (2 draws, rtol 0.01, atol 0.01, weights by_position)
$ smforge pipeline --ids my_linear_relu
```

The `-` is its split: a problem outside `corpus/splits.yaml` is in neither `dev` nor `eval`, so
select it with `--ids`. The family picks which [knowledge base](knowledge-base.md) patterns
its prompts get.

## 7. See what the LLM is told

```
$ smforge kb show memory_access --families gemm --dsl cuda
## Constraints

- **launch_error_check** (critical): Check every kernel launch: call C10_CUDA_KERNEL_LAUNCH_CHECK()
  in the binding after the launcher returns (or return cudaGetLastError() from it). ...
```

That is the slice of the knowledge base a `memory_access` stage on a GEMM sees: the rules, then
the patterns, each verified by a kernel pair on the GPU. More in [knowledge base](knowledge-base.md).
