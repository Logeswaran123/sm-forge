# sm-forge: build plan

sm-forge takes a correct Triton kernel and a shape spec, and returns a faster kernel that passes
the same correctness checks on an NVIDIA GPU. An LLM works through a sequence of optimization
stages. Every candidate is compiled, checked against a reference and timed on the GPU before it
is kept. The design follows Xe-Forge (IntelLabs, arXiv:2605.26118), which does the same for Intel
Arc GPUs.

The first target is the GeForce RTX 5080 in this machine (sm_120). The knowledge base is gated by
compute capability, so sm_80, sm_89, sm_90 and sm_100 parts can be added without forking the code.

## 1. What Xe-Forge does

Paper: *Xe-Forge: Multi-Stage LLM-Powered Kernel Optimization for Intel GPU*, Spoczynski et al.,
April 2026. Repository: github.com/IntelLabs/Xe-Forge, v0.4.0, Apache-2.0, about 18k lines of
Python under `src/xe_forge`.

### Pipeline

1. The analyzer (an LLM) reads the kernel and returns typed issues from a taxonomy of 30+
   `IssueType`s (`dtype_float64`, `unfused_elementwise`, `missing_autotune`, `open_ended`, ...).
   Each issue has a severity from 1 to 5, a suggested fix and a speedup estimate.
2. A fixed table routes each issue type to one stage. Stages with no issues are skipped.
3. The planner (an LLM) orders the remaining stages. Its output is filtered, then corrected
   against hard dependencies in `planner.py`: `algorithmic < dtype_fix < fusion < device_specific
   < autotuning`, `memory_access < tensor_descriptors < device_specific`, `discovery < dtype_fix`.
   The fallback order is algorithmic, discovery, dtype_fix, fusion, memory_access,
   tensor_descriptors, persistent_kernel, device_specific, autotuning. (`tensor_descriptors` was
   `block_pointers` until Triton deprecated `tl.make_block_ptr`.)
4. Each stage runs a CoVeR agent (Chain of Verification and Refinement, `agents/cover.py`). The
   LLM writes a thought and a full kernel. The harness runs the verification tool on it. On
   failure the error is appended to the trajectory and the LLM tries again, up to T iterations
   (5 in the paper, 3 in the reproduction config). When the budget runs out, a fallback
   extractor produces one more candidate. It must pass the same checks, or the stage returns its
   input unchanged. On context overflow the oldest iteration is dropped.
5. The kernel is re-analyzed after each stage. The whole sequence can run k times and keep the
   fastest result (best-of-k, default 1).

### Verification cascade

1. Syntax: `ast.parse`.
2. Structure: imports, `@triton.jit`, `class Model`, power-of-two `num_warps`, block dims ≤ 256.
3. Correctness: seeded init, weights copied with `load_state_dict`, cloned inputs, `torch.allclose`
   at rtol 1e-2 and atol 1e-5. NaN is rejected. Inf is rejected unless the original has it.
4. Performance: the candidate must be faster than its input.

### Timing

AI-bench: 200 warmup iterations, 100 timed, a 256 MB L2 flush before each, device events, min and
max trimmed, mean reported. The repository later added interleaved A/B rounds with alternating
order and a 2% minimum-effect band, after an A/A check measured about 1% between identical builds
(`core/executor.py: time_interleaved`).

### Knowledge base

YAML files of constraints (hard rules with a severity and wrong/correct snippets) and patterns
(before/after transformations tagged by stage), plus 19 unoptimized/optimized example pairs used
as few-shot material. The paper counts 84 entries. Each stage's prompt gets the entries for that
stage. The README says the examples are drawn from KernelBench, the suite the paper reports on.

### Engines and tooling

- DSPy engine: the automated pipeline above.
- Coordinator: a ReAct agent that picks stages itself (`agents/coordinator.py`).
- Claude Code engine: writes a workspace (`CLAUDE.md`, `config.yaml`, KB, slash commands) and
  drives it through the `xe-forge-skill` CLI (`analyze`, `validate`, `benchmark`,
  `trial init|save|status|finalize`, `profile`). The skill CLI is the only permitted source of
  timings, and results go into a trial tree (`core/trial_manager.py`).
- Tile search: LLM-proposed tile shapes for CUTLASS-SYCL GEMM and Flash Attention templates.
- Host integration: `--kernel-repo`, `--external-benchmark`, `--lessons`, `--dataset-record`.

### Results (Arc Pro B70, about 160 TFLOPS FP16, 608 GB/s, GPT-5.4)

- 1.17x geomean over PyTorch eager on 97 KernelBench Level 2 kernels. 67% improved, 9 above 5x,
  maximum 82x.
- Flash Attention forward: 2 to 13.3x over the seed kernel on 16 shapes, no regressions.
- Conv and ConvTranspose families regress (0.5 to 0.8x on several).
- Seeds came from KernelFalcon (KernelAgent).
- Typical cost: 10 to 20 LLM calls per kernel, $0.50 to $2.00.
- The paper attributes results above peak to FLOP reduction, not throughput. It names per-stage
  rollback as the most useful next step.

### NVIDIA support in Xe-Forge today

`--device cuda` and `uv sync --extra nvidia` exist, but:

- `knowledge_base/triton/cuda/cuda_optimizations.yaml` contains `patterns: []` and
  `constraints: []`.
- `core/device_query.py: _set_cuda_recommendations` branches on `major >= 9` and returns Hopper
  settings (128x256x64 tiles, 8 warps). sm_120 has major 12, so the RTX 5080 gets Hopper tiles.
  This part has 99 KB of shared memory per block and no wgmma.
- Prompts, the validator and the stage list are XPU-specific: `grf_mode`, 32 warps by default,
  tensor descriptors described as 2D block IO.
- Candidates are compiled and run inside the pipeline process. On CUDA an illegal address is a
  sticky error: every later CUDA call in that process fails, so one bad candidate ends the run.

The parts to carry over:

- the structure: analyze, route, plan, CoVeR per stage, re-analyze;
- the verification cascade and the interleaved timing;
- the oracle rules in `docs/CORRECTNESS.md` and the trial ranking;
- the rule that only the harness produces timings.

The hardware knowledge has to be written from scratch.

## 2. Target hardware

Measured on the development machine with `torch.cuda.get_device_properties` and `nvidia-smi`
(torch 2.11.0+cu128, triton 3.6.0; CUDA 12.8 toolkit with ncu, nsys, compute-sanitizer,
nvdisasm):

| Property | Value |
|---|---|
| GPU | GeForce RTX 5080, compute capability 12.0 (sm_120) |
| SMs | 84 |
| Memory | 16 GB GDDR7 |
| L2 | 64 MB |
| Shared memory | 100 KB per SM, 99 KB (101376 B) per block opt-in |
| Registers | 64K 32-bit per SM |
| Max SM clock | 3090 MHz |
| Driver | 617.42, Windows host, WSL2 guest |

Measured in stage 0 (`smforge device --all`, saved to `hardware/rtx5080.yaml`):

| Item | Result |
|---|---|
| fp16, bf16, tf32 `tl.dot` | HMMA |
| fp8 `tl.dot`, `tl.dot_scaled` (e4m3) | QMMA |
| `num_stages > 1` | cp.async (LDGSTS) |
| Tensor descriptors, host and device side | TMA (UTMALDG, UTMASTG) |
| `tl.range(..., warp_specialize=True)` | compiler error, `PassManager::run failed` |
| DRAM bandwidth | 885 GB/s read, 755 GB/s copy |
| cuBLAS peak | fp16 116, bf16 109, tf32 53, fp32 38, fp8 234 TFLOPS |
| A/A timing noise (p20) | under 1%; suggested min effect 1.7% |
| nsys | works |
| ncu | works once GPU performance counters are enabled on the Windows side; `ERR_NVGPUCTRPERM` before |
| compute-sanitizer | works once the WDDM debugger interface is enabled (`EnableDebuggerInterface.bat`) |

Cluster launch and clock locking were not tested. Clock locking needs root.

## 3. Intel to NVIDIA mapping

| Xe-Forge (Intel Xe) | sm-forge (NVIDIA) |
|---|---|
| XMX / DPAS | Tensor cores: mma.sync (sm_80+), wgmma (sm_90a), tcgen05 (sm_100a) |
| Subgroup | Warp (32 lanes) |
| SLM | Shared memory, budget per block from the device query |
| GRF mode 128/256 | Register budget: `maxnreg`, `num_warps`, occupancy |
| 2D block IO via tensor descriptors | TMA via tensor descriptors (sm_90+); `cp.async` pipelining via `num_stages` (sm_80+) |
| 32 warps by default | Usually 4 or 8 warps, tuned |
| Xe cores | SMs; persistent grid = SM count × CTAs per SM |
| `xpu-smi`, `torch.xpu` | NVML, `nvidia-smi`, `torch.cuda` |
| VTune, unitrace | Nsight Compute (`ncu`), Nsight Systems (`nsys`) |
| SPIR-V check (DPAS vs FMA count) | SASS check (tensor-core MMA vs FFMA, LDGSTS, UTMALDG, spills) |
| CUTLASS-SYCL tile search | CUTLASS / CuTe tile search |

## 4. Design decisions

1. **Triton first.** CUDA C++ comes in stage 7. Python 3.11+, uv, torch cu128, Triton pinned
   (3.6.0 works on sm_120).

2. **Process isolation.** Every candidate compiles and runs in a worker subprocess with a timeout.
   A worker that hits a CUDA error is discarded and a new one started. Reference outputs for each
   kernel and input draw are computed once and stored as `.pt`, so checking a candidate costs one
   worker run.

3. **Fixed harness and a narrow edit surface.** The LLM returns a kernel file. The harness, the
   spec and the reference are read-only. The paper describes generated kernels that call PyTorch
   through dynamic imports to pass checks. Three guards against that:
   - AST rule: `Model.forward` and its helpers may call only allocation, view and layout ops from
     torch. The allowlist can be extended per kernel.
   - Launch audit: `torch.profiler` records the CUDA kernels one forward launches. A candidate must
     launch at least one Triton kernel. It may not launch a vendor kernel (cuBLAS, cuDNN, aten
     CUTLASS) that the baseline does not launch.
   - Output caching: correctness runs two input draws in the same process. A module that returns
     a stored result fails the second draw.

4. **Facts before the LLM.** The analyzer gets numbers measured from the compiled kernel:
   - registers, spills and shared memory per kernel (`CompiledKernel.n_regs`, `.n_spills`,
     `.metadata.shared`);
   - grid size and waves on 84 SMs;
   - SASS instruction mix (tensor-core MMA vs FFMA, cp.async, TMA), from `.asm["cubin"]` through
     nvdisasm;
   - resident CTAs per SM from the driver's occupancy calculator;
   - arithmetic intensity from the spec's flop and byte formulas, against the measured ridge point;
   - an ncu Speed of Light summary, when counters are available.

   The LLM classifies issues from these instead of guessing from source.

5. **Arch-gated knowledge base.** Each entry has `arch: {min: 80, max: null}` and optional
   `requires: [tma, fp8_mma, ...]`. The loader filters by the detected device and its feature
   probes. Each entry has a runnable before/after pair and a speedup measured on a named GPU, and
   CI runs the pairs.

6. **Per-stage rollback.** A stage result is kept only if it passes correctness and beats the
   stage input by more than the min-effect band in an interleaved A/B measurement. Otherwise the
   stage input carries forward. A regression is never finalized.

7. **LLM layer.** Call the Anthropic Python SDK directly, behind a small `LLMClient` interface. Do
   not use DSPy. CoVeR is a loop of about 200 lines, and calling the SDK directly keeps prompt
   caching, effort and output schemas under our control.
   - Default model `claude-opus-5-5` for every role. Effort is set per role: analyzer `high`,
     optimizer `high`, planner `low`, fallback extractor `medium`.
   - The analyzer and planner use structured outputs (`client.messages.parse` with Pydantic
     models).
   - The optimizer returns the kernel as a fenced code block. No tool calling is needed, because
     the harness verifies every candidate itself.
   - Two API constraints differ from Xe-Forge's config. The Messages API has rejected
     `temperature` since 2026-10-21 (Xe-Forge sets `LLM_TEMPERATURE=1`). Opus 5.5 rejects forced
     `tool_choice`.
   - Prompt layout for caching: a stable prefix (role instructions, device facts, the stage's KB
     slice) marked with `cache_control`, then the kernel and the trajectory.
   - Log `usage` on every call, including `cache_read_input_tokens`. Enforce a per-kernel dollar
     cap.

8. **Two engines.** A pipeline engine (stages 4 and 5) and a Claude Code workspace engine
   (stage 6). Both go through the same harness via `smforge-skill`.

9. **Corpus split.** KB authoring and prompt tuning see only the dev split. The eval split is
   frozen before stage 3 starts.

10. **Timing statistic.** A run's time is a low quantile of its samples (p20 by default), not the
    median. Interference from other work on the GPU only adds time. On this WSL2 machine the
    Windows desktop slowed up to 40% of the samples of a 1.2 ms matmul: the medians of two
    identical A/A arms differed by 16%, their p20s by under 0.1%.

11. **Fitting KernelBench to the GPU.** A problem is translated to a spec by reading its source as
    an AST; its code is never imported to build the spec. The upstream shapes stay as the
    `kernelbench` variant. `bench-gpu` halves the batch (else the first input's leading dim) until
    the float16 reference fits 4 GiB and 5 ms, then takes the first of float16, bfloat16 and
    float32 whose output is finite and within that dtype's tolerance of float32. `ci` is
    `bench-gpu` at batch 3.

12. **Oracle precision.** The worker turns off TF32 and reduced-precision split-K reductions in
    cuBLAS, so a reference accumulates in float32 whatever its dtype. Eager baselines run under the
    same settings. With PyTorch's defaults the bf16 reference of a 3×524288×256 matmul was 5% off
    float32. For correctness, a float32 problem's PyTorch reference runs in float64: eager float32
    was 0.032 off float64 on a ConvTranspose2d + GroupNorm problem at batch 512, where a Triton
    kernel was 8e-6 off.

13. **Seeds through Claude Code.** `smforge seed` (option B) runs CoVeR with correctness-only
    verification. The LLM backend is the Messages API or headless Claude Code (`claude -p`, no
    tools, in an empty directory); stage 2 used Claude Code.

14. **WSL2 workarounds.** The PyTorch profiler stops recording CUDA activity about 40 s after a
    process first profiles, so each launch audit runs in a fresh worker and an empty profile is an
    error. An out-of-memory attempt can spill into system memory and slow the worker for the rest
    of its life, so each imported problem is fitted in a fresh worker.

## 5. Repository layout

```
sm-forge/
  pyproject.toml              # uv, hatchling; extras: cu128, dev
  .env.example
  src/smforge/
    cli.py                    # smforge: optimize, eval, device, seed
    skill_cli.py              # smforge-skill: analyze, validate, check, bench, facts, trial, profile
    config.py
    models.py                 # pydantic: Issue, Analysis, StageResult, Trial, RunReport
    spec.py                   # spec.yaml loader, flop/bytes formulas, variants
    render.py                 # text output for the CLIs
    device/
      query.py                # torch, driver and nvidia-smi properties
      probes.py               # feature probes: compile, check SASS and output
      kernels.py              # probe and bandwidth kernels
      roofline.py             # measured bandwidth and matmul peaks
      noise.py                # A/A timing noise
      tools.py                # nvidia-smi; ncu, nsys, compute-sanitizer checks
      profile.py              # hardware/<gpu>.yaml
    harness/
      runner.py               # parent side: worker lifecycle, timeouts, restart after CUDA errors
      worker.py               # subprocess entry: build, run, check, time
      workload.py             # kernel loading, inputs and Model construction per variant
      source.py               # AST view of a kernel file
      static.py               # Triton static validator
      audit.py                # forward-path rules, launch audit
      tolerance.py
      correctness.py          # output comparison, weight copy, reference cache
      timing.py               # events, L2 flush, quantile, interleaved A/B, verdict
      launches.py             # kernels one forward launches, from the profiler
      facts.py                # regs, spills, smem, SASS mix, occupancy, waves
      ncu.py                  # ncu counters, section profiles and diffs
      sanitizer.py            # compute-sanitizer wrapper
    kb/
      schema.py               # constraints, patterns, examples, measurements
      loader.py               # arch and probe filter, stage slices, token budget
      verify.py               # runs each entry's kernel pair on the GPU
    llm/
      client.py               # SDK wrapper: caching, structured output, usage, cost cap, replay
      prompts/                # jinja templates per role and stage
    agents/
      analyzer.py
      planner.py
      cover.py
      stages.py               # stage registry, routing table, dependencies
    pipeline.py
    trials.py                 # trial tree
    lessons.py
    claude_ws/                # Claude Code workspace generator and templates
    dsl/
      triton.py
      cuda_cpp.py             # stage 7
    tile_search/              # stage 7
  knowledge_base/
    triton/constraints.yaml
    triton/patterns/<stage>.yaml
    triton/checks/<constraint>/{wrong.py,correct.py}
    triton/examples/{index.yaml,measured.yaml,<example>/{before.py,after.py}}
    cuda/                     # stage 7
  corpus/
    <id>/{reference.py,kernel.py,spec.yaml}
    splits.yaml               # dev / eval
  hardware/
    rtx5080.yaml              # stage 0 output
  runners/
    test_kb_examples.py
  scripts/                    # roofline plots, report generation
  tests/
  docs/
```

## 6. Stages

Each stage lands as several focused commits. Exit criteria are checked before the next stage
starts.

### Stage 0: environment and hardware baseline

- `pyproject.toml` with uv. Pin torch 2.11.0+cu128 and triton 3.6.0.
- `smforge device` writes `hardware/rtx5080.yaml`: the properties in section 2, plus feature probes.
  Each probe compiles a small Triton kernel and checks its SASS. The probes cover:
  - tensor-core MMA for fp16/bf16 `tl.dot`;
  - fp8 `tl.dot`;
  - TMA from `tl.make_tensor_descriptor`;
  - `tl.dot_scaled`;
  - warp specialization.
- Measured roofline:
  - DRAM bandwidth from copy and read-only kernels at sizes above L2, and L2 bandwidth;
  - tensor throughput per dtype from a cuBLAS/cuBLASLt GEMM sweep (fp16, bf16, tf32, fp8 if
    available);
  - FP32 FMA peak.

  Store ridge points. These replace spec-sheet numbers everywhere downstream.
- Timing noise:
  - Run an A/A test (the same kernel in both arms) on a small kernel, a memory-bound kernel and a
    compute-bound kernel, using stage 1's timing code. Set `min_effect` from the spread.
  - Check whether `nvidia-smi -lgc` works under WSL2.
  - Measure the effect of the Windows desktop sharing the GPU.
- Profiler access:
  - Run `ncu --section SpeedOfLight` on a toy kernel. Under WSL2 this needs GPU performance
    counter access enabled on the Windows side.
  - Check that `nsys` and `compute-sanitizer` run.

Exit: `hardware/rtx5080.yaml` committed with measured peaks and probe results. The A/A spread is
known. There is a written list of what works under WSL2 and what does not.

### Stage 1: harness, no LLM

Kernel contract, KernelBench-compatible and the same as Xe-Forge's:

- `kernel.py`: `class Model(nn.Module)`, `get_inputs()`, `get_init_inputs()`;
- `reference.py`: the PyTorch `Model`;
- `spec.yaml`: `inputs`, `inits`, and variants (`ci`, `bench-gpu`, ...) with `dims`, `dtype`,
  `flop`, `bytes`, and optional `rtol`/`atol`.

Xe-Forge's test kernels then port with device changes only.

Work:

- Spec loader with formula evaluation.
- Worker subprocess and pool: JSON over pipes, timeout, restart after a CUDA error.
- Correctness:
  - seeds, weight copy, cloned inputs, two input draws;
  - tolerances by dtype; precedence CLI > spec > env > default;
  - NaN and Inf rules;
  - diagnostics: max abs/rel error, share of mismatched elements, first bad index.
- Timing:
  - device events, a 256 MB flush buffer (4x L2), and a dummy matmul queued ahead of the start
    event so launch latency is not timed (as Xe-Forge does);
  - warmup, trimmed statistics;
  - interleaved A/B rounds with alternating order, medians and per-round ratios;
  - a verdict of `faster`, `slower` or `parity` against `min_effect`;
  - an optional CUDA graph mode for launch-bound kernels, reported separately.
- Two time measures per run: forward wall time (events around `forward`) and kernel-only time (sum
  of device kernel durations from the profiler). Many seeds cast weights or synchronize inside
  `forward`. Removing that is a real speedup, but it is not a kernel speedup, and reporting both
  keeps the difference visible.
- Static validator:
  - syntax, imports, `@triton.jit`, `class Model`;
  - power-of-two `num_warps` and `BLOCK_*`;
  - `tl.dot` operand dims ≥ 16;
  - estimated shared memory per config ≤ device opt-in;
  - no `.item()`, `.cpu()` or `torch.cuda.synchronize()` in `forward`.
- Audit: the AST rule and launch audit from section 4.
- Facts: regs, spills, smem, SASS mix, waves, arithmetic intensity.
- CLI: `smforge-skill validate | check | bench | facts`.
- Tests:
  - unit tests per module;
  - integration tests on hand-written kernels, including broken ones: wrong mask, out-of-bounds
    write (must kill the worker and recover), a call to `torch.matmul`, an output cache, a host
    sync in `forward`.

Exit: hand-written kernels repeat within the A/A band across runs. Every broken kernel is rejected
with a message that names the cause.

### Stage 2: corpus and baselines

- Port KernelBench Level 2 (all 100) and a selection of Level 1 into the corpus format, with
  fp16/bf16 variants, and fp32 where the problem needs it. Port Xe-Forge's `test_kernels/` and
  `examples/vllm/`, replacing `torch.xpu` with `torch.cuda`.
- Add inference kernels:
  - RMSNorm, fused add + RMSNorm, LayerNorm, RoPE, SwiGLU, softmax;
  - Flash Attention forward (prefill shapes), decode and paged attention;
  - fused MoE;
  - FP8 GEMM with per-tensor and per-row scales.

  Use real model shapes sized for 16 GB. Check the license of any kernel imported from vLLM or
  SGLang before committing it.
- Seed kernels. Pick one of:
  - Option A: KernelAgent (meta-pytorch/KernelAgent) on the reference, as the paper did.
  - Option B: `smforge seed`, which runs the CoVeR loop with correctness-only verification and a
    prompt asking for the straightforward kernel.

  Freeze the seeds. A seed in the eval split is never regenerated after results are reported.
- Baselines per variant: eager, `torch.compile` (default and `max-autotune`), seed kernel. Write
  `results/baselines.csv` with torch, triton and driver versions and the git hash.
- Split `corpus/splits.yaml` into dev and eval, stratified by family (GEMM, matmul, conv,
  convT, norm, attention, MoE, reduction). Freeze the eval split.
- Roofline script: each kernel plotted at its arithmetic intensity against the stage 0 ceilings.

Exit: at least 120 problems with passing seeds and measured baselines. The split is committed.

Result (2026-10-27): 138 problems: 12 inference, all 100 of Level 2 and 26 of Level 1. Dev holds
96 and eval 42. Every problem has a seed from Claude Code (Opus 5.5, effort high). Each of the 126
KernelBench seeds passed on its first attempt, except l1_006, l2_011 and l2_014. Those three
failed four attempts against an inaccurate oracle and passed first time after the fixes in
decision 12. Seeding cost $17.12 at list price. Geometric-mean speedup over eager at `bench-gpu`:

| Problems | `torch.compile` | Seed | Seeds faster than eager | Seeds faster than compile |
|---|---|---|---|---|
| Inference (12) | 2.07× | 4.12× | 11 | 9 |
| Level 1 (26) | 1.49× | 1.47× | 14 | 10 |
| Level 2 (100) | 1.55× | 1.24× | 60 | 41 |

Seven Level 2 seeds rewrite their problem algebraically: a GEMM or transposed convolution
followed by a sum or a global pool becomes sums of the input and of the weights, recomputed on
every call. Against the reference's FLOP count these seeds exceed the tensor peak, up to 8232
TFLOPS for l2_077. Not done: Xe-Forge's test and vLLM kernels, and `max-autotune` baselines.

### Stage 3: knowledge base

Schema (YAML, validated with pydantic in `kb/schema.py`):

```yaml
# constraints.yaml
constraints:
  - id: smem_budget_per_config
    severity: critical
    stages: [tensor_descriptors, device_specific, autotuning]   # empty: every stage
    arch: {min: 80}
    requires: []                  # device probes that must pass
    rule: "num_stages * (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * elem_bytes <= the opt-in limit"
    wrong: |
      BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
    correct: |
      BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    check: {problem: l1_002_standard_matrix_multiplication, expect: error, match: shared memory}
# patterns/<stage>.yaml
patterns:
  - id: split_k_for_small_mn
    stage: persistent_kernel
    applies_to: [gemm, matmul]
    summary: ...
    how: ...
    example: gemm_split_k         # examples/gemm_split_k/{before.py,after.py}
```

A constraint's `checks/<id>/wrong.py` must fail as `check.expect` says (`static`, `error`,
`mismatch` or `slower`) and its `correct.py` must pass the candidate checks. A pattern's example
runs on a dev-split problem: `after.py` must pass the candidate checks and beat `before.py` in an
interleaved A/B comparison. Speedups are written to `examples/measured.yaml` by
`smforge kb verify --write`, never by hand. Prompts do not show speedups. Xe-Forge's Claude
template tells the agent to ignore them, and this KB leaves them out.

First content. Each entry is verified on the RTX 5080 before it goes in.

Constraints:

- Shared memory per config within the device's opt-in limit (99 KB per block here).
- `tl.dot` operand shapes ≥ 16 and powers of two.
- fp32 `tl.dot` defaults to TF32 (`input_precision`). Against an fp32 reference with tight
  tolerance, use `input_precision="ieee"` or change the dtype policy explicitly. PyTorch's matmul
  does not use TF32 by default.
- int64 offsets when `numel` can exceed 2^31.
- Register budget: `num_warps * 32 * regs_per_thread ≤ 65536`. Spills are reported by facts.
- No host syncs, `.item()` or `.cpu()` in `forward`. No per-call weight casts or transposes: do them
  once and cache them on the module.
- Tensor descriptors (TMA): 16-byte aligned base, non-last strides that are multiples of 16 bytes.
- Autotune: `key` must include the shape arguments that change the best config. Kernels that
  accumulate in place need `reset_to_zero` or `restore_value`.
- fp32 `atomic_add` makes results order-dependent; check against the tolerance.
- GROUP_M swizzling needs a 1D grid.

Patterns by stage:

| Stage | Patterns |
|---|---|
| algorithmic | algebraic rewrites (reduce before a GEMM when the reduction is linear, fold constant scales); review and reuse Xe-Forge's `common/algorithmic_patterns.yaml` |
| dtype_fix | fp16/bf16 I/O with fp32 accumulate; fp8 e4m3 with scales only when the spec tolerances allow; bf16 when fp16 range is at risk |
| fusion | GEMM epilogue fusion (bias, activation, scale, residual); a reduction in the GEMM epilogue when N fits one tile; online softmax; no fusion that pushes registers into spills |
| memory_access | contiguous dim on the fastest axis; `tl.multiple_of` / `tl.max_contiguous` for vector width; masks only on tails (`EVEN_K` specialization); `eviction_policy`; contiguity in the wrapper; channels-last for conv |
| tensor_descriptors | TMA loads via `tl.make_tensor_descriptor` on sm_90+, and on sm_120 if the probe passes; `num_stages` pipelining on sm_80+ |
| persistent_kernel | persistent GEMM with `NUM_SMS` from the device; split-K / stream-K for small M·N with large K; tail-wave analysis (tile count mod SM count) |
| device_specific | tile shape tables per arch from measured sweeps; `num_warps`, `num_stages`, `maxnreg`; GROUP_SIZE_M; warp specialization where the probe passes |
| autotuning | config space generated from the constraints and pruned by smem and register budget (`prune_configs_by`); at most 12 configs; correct keys |

Examples: 15 to 20 unoptimized/optimized pairs, taken from the dev split only.

Loader: filter by arch and probes, slice by stage, rank entries by how well their `applies_to`
matches the kernel's op families, and cap each stage at a token budget (start at 6k).

Exit: every entry verified on the RTX 5080 with a pair from the dev split, and every stage has at
least one pattern. `runners/test_kb_examples.py` passes. The plan first asked for 60 entries; it
was lowered to this after the result below.

Result (2026-10-27): 39 entries, 18 constraints and 21 patterns, every one verified on the RTX
5080 with a pair from the dev split. All eight stages have at least one pattern. Pattern speedups range from 1.04x (GEMM epilogue fusion, where the GEMM
dominates) to 27.8x (reducing before a linear op). These measured at parity or worse on sm_120
and are left out:

| Candidate | Measured |
|---|---|
| GROUP_M swizzle on a 1024x8192x8192 GEMM | 1.00x; the 64 MB L2 already holds the reuse |
| EVEN_K specialization (no K masks) | 0.99x |
| TMA tensor descriptors instead of pointer loads | 1.00x at 64x64x64, 0.80x at 128x128x32 |
| 128x128 GEMM tiles with 8 warps | 0.88x against 64x64x64 with 4 warps |
| More warps for a 4096-wide RMSNorm row | 1.00x |
| Fused add + RMSNorm instead of two kernels | 0.94x; the 32 MB intermediate stays in L2 |
| Split group-norm statistics over 112 groups | 1.00x |
| Unmasked key blocks below the causal diagonal | 1.00x to 1.02x |
| One program per GQA query head in paged decode | 1.30x over grouping, but 924 us against 777 us for split-KV |

Two constraints could not be shown failing on these problems and are left out: a softmax tail
padded with 0 instead of -inf shifts the sum by about 1% on 128256-wide rows, inside bf16
tolerance, and eval-mode batch norm with fresh running statistics matches batch statistics on
randn inputs.

### Stage 4: LLM client and CoVeR

- `llm/client.py`:
  - model and effort per role from config;
  - streaming for long outputs;
  - structured outputs via `messages.parse`;
  - prompt caching on the stable prefix, with `cache_read_input_tokens` checked so a silent miss
    shows up in the log;
  - per-call cost log and a per-kernel cost cap;
  - SDK retries for 429 and 5xx;
  - handling of `stop_reason == "refusal"`.
- Record and replay: every request and response stored as JSONL keyed by prompt hash. Tests run
  from recordings without network access.
- CoVeR:
  - each iteration produces a thought and a full kernel;
  - verification order: syntax, static, audit, compile in a worker, correctness, facts, timing
    against the stage input;
  - feedback is short: the first compiler error with its source line, correctness diagnostics,
    the timing verdict, fact deltas (`regs 96 -> 168, spills 0 -> 24`);
  - T = 3 by default, configurable;
  - fallback extractor;
  - truncation by dropping the oldest iterations;
  - every failed candidate saved with stage, iteration and error class.
- Output parsing takes the last fenced Python block. The candidate is rejected if it changes
  `get_inputs`, `get_init_inputs` or the `Model.__init__` signature.
- First stage to wire up: autotuning, which has the narrowest edit surface.

Exit: the autotuning stage runs over the whole dev split with no correctness failures in finalized
kernels. Replay tests pass offline. Accept rate and cost per kernel are recorded.

Built in v0.4.0 as listed, with these differences:

- Structured outputs and the fallback extractor move to stage 5. The optimizer replies with one
  fenced block and needs neither; the analyzer and planner are the first roles with structured
  replies.
- The cost cap is per run (`--budget-usd`), not per kernel.
- Replay keys mask decimals and integer rates (GB/s, TFLOPS): prompts quote timings, which
  differ between runs.
- Order of checks: static checks and source audit, interface, correctness on `ci` and `bench-gpu`
  (compile errors surface here), launch audit against the input's profile, A/B timing. Facts are
  collected only for candidates that are correct but not faster, to report deltas.
- The launch audit fails closed. A kernel is allowed if Triton launched it (defined in the module
  or seen by Triton's runtime), if it moves memory, or if the input launches it too; anything
  unclassified is an error, and `nvjet` counts as cuBLAS. All 138 seeds pass.

Result (2026-10-28): the autotuning stage over the 96 dev problems, starting from the seeds, with
up to 3 candidates each through Claude Code (Opus 5.5, effort high):

| | |
|---|---|
| Accepted | 77 of 96, 63 on the first candidate |
| Kept the input | 19; every candidate at parity or slower |
| Candidates | 152: 77 accepted, 72 parity, 3 slower; none failed the static, interface, correctness or launch checks |
| Speedup over the seed, geomean | 1.35x over all 96 (1.26x without l2_065), 1.46x over the accepted |
| Cost at list price | $23.97 in 152 calls: $0.25 per problem, $0.31 per accepted kernel |

The conv families gain most (conv2d 2.11x, convtranspose2d 2.00x, conv3d 1.62x geomean); GEMM
and matmul seeds were already near cuBLAS and gain 1.06x to 1.08x. The largest, l2_065 at 1333x,
is a seed whose 64-wide, 4-warp, 3-stage config held the conv kernel to 40 registers with 2972
spills; the autotuned 32-wide, 1-warp, 1-stage config uses 154 registers and none. All 77 accepted
kernels pass correctness on `ci` and `bench-gpu` and the launch audit again in fresh workers,
where the autotuner picks its config anew. Per-problem rows are in `results/stages.csv`.

### Stage 5: analyzer, planner, all stages

- Issue taxonomy. Start from Xe-Forge's `IssueType` and drop the XPU-only types (`missing_grf_mode`,
  `missing_packed_transpose`). Add:

  | Issue | Detected from |
  |---|---|
  | `no_tensor_core_mma` | SASS shows an FFMA loop where a dot was intended |
  | `tf32_precision_risk` | fp32 dot with default precision against a tight tolerance |
  | `register_spill` | `n_spills > 0` |
  | `low_occupancy` | regs or smem limit CTAs per SM |
  | `smem_over_budget` | config estimate above the device limit |
  | `missing_pipelining` | `num_stages = 1`, no cp.async in SASS |
  | `missing_tma` | sm_90+ or probe-enabled sm_120, no UTMALDG |
  | `tail_wave` | small remainder of tiles mod SMs |
  | `grid_underfill` | fewer CTAs than SMs |
  | `split_k_opportunity` | small M·N, large K |
  | `launch_bound` | many short kernels per forward |
  | `uncoalesced_access` | ncu memory sectors per request, or the analyzer |
  | `bank_conflict` | ncu shared-memory metrics |
  | `fp8_opportunity` | spec tolerance allows it, compute-bound GEMM |
  | `host_sync_in_forward`, `per_call_weight_prep` | AST |

- Routing table from issue to stage, with a test that every issue type routes somewhere.
- Analyzer: structured output. Input is the kernel, the reference, the spec summary, the facts and
  the roofline position. A deterministic pre-analyzer emits the fact-backed issues (spills, waves,
  FFMA dot, host sync), and the LLM cannot drop them.
- Planner: LLM ordering, Xe-Forge's hard-dependency correction, skip logic, deterministic fallback.
- One prompt template per stage, with that stage's KB slice.
- Re-analysis after each stage, per-stage rollback, best-of-k.
- Discovery stage for `open_ended` proposals. A proposal needs a validity argument and a speedup
  estimate, and goes through the same verification.

Exit: the full pipeline runs over the dev split, and no regression is finalized. The run report
has accept rates and cost per stage.

Built in v0.5.0, with these differences from the list above:

- Issue types: `missing_tma` and `fp8_opportunity` are left out (TMA measured at parity on sm_120,
  and no fp16 problem's tolerance admits fp8); `untuned_config` (no `@triton.autotune`) and
  `per_call_weight_prep` are added. `uncoalesced_access` and `bank_conflict` come from Nsight
  Compute: global loads using under 24 of 32 bytes per sector, and more than 0.25 shared-memory
  bank conflicts per load wavefront. Loads through cp.async do not count as LSU loads, so a
  pipelined kernel has no load figure.
- Structured replies for the analyzer and planner: the API sends the pydantic model's schema as
  `output_config.format`, Claude Code as `--json-schema`.
- Re-analysis after each accepted stage is deterministic (new measurements, fact-backed issues
  recomputed); the analyzer LLM runs once per problem. Best-of-k is not built.
- `open_ended` is a stage, run before autotuning, for the analyzer's `open_ended` proposals.
- Every config of an autotuned kernel is checked for correctness, not only the one the autotuner
  picks. The check rebuilds draw 0 rather than keeping it: keeping its tensors alive moved the
  allocator's layout, and the unmasked tail store in the `mask_tail_stores` KB check stopped
  showing as a mismatch. The dev run below used the version that kept it.
- Three static rejections, each a KB constraint, added after the first dev run produced kernels
  that passed every output check: a `tl.dot` accumulating in fp16 or bf16 (l2_076, 1.68x), a
  `while` loop polling an atomic or volatile load, which waits on other programs and deadlocks when
  they are not resident (inf_softmax_vocab), and any assignment to `self` in the forward path
  (l2_014 cached the weights' column sums on the first call, 9.65x). The affected problems were
  rerun.
- Cost controls: `--max-stages` caps the stages per problem besides autotuning, and the optimizer's
  effort is set per role. Without them a problem cost $5 to $8.
- Results go to `results/pipeline.csv` (one row per problem) and `results/pipeline_stages.csv`
  (one row per stage run).

Result (2026-10-30): the pipeline over the 96 dev problems, starting from the seeds, through Claude
Code (Opus 5.5, effort medium, planner low), at most 2 stages plus autotuning and up to 2
candidates per stage:

| | |
|---|---|
| Improved | 80 of 96; the other 16 kept the seed, every candidate at parity, slower or wrong |
| Regressions finalized | none |
| Speedup over the seed | geomean 2.55x, median 1.25x |
| Against torch.compile (`bench-gpu`) | final faster on 87 of 96, geomean 2.36x; the seeds were faster on 41, 0.93x |
| Candidates | 393: 157 accepted, 157 parity, 66 slower, 13 wrong on `ci`; none failed the static, interface or launch checks |
| Cost at list price | $102.58, $1.07 per problem; $10.19 of it analyzer and planner |

| Stage | Accepted | Geomean of accepted | Cost |
|---|---|---|---|
| autotuning | 32 of 96 | 1.11x | $35.22 |
| open_ended | 33 of 40 | 1.93x | $13.93 |
| memory_access | 25 of 36 | 1.45x | $13.84 |
| persistent_kernel | 19 of 25 | 1.15x | $8.73 |
| algorithmic | 20 of 21 | 11.18x | $4.40 |
| fusion | 14 of 20 | 1.22x | $7.47 |
| tensor_descriptors | 8 of 10 | 1.13x | $5.06 |
| device_specific | 4 of 6 | 1.15x | $2.91 |
| dtype_fix | 2 of 2 | 1.17x | $0.83 |

By family: Level 2 3.38x (70 problems), inference 1.26x (10), Level 1 1.15x (16). The largest
gains are on seeds that take 17 to 136 ms: l2_065 1702x, l2_096 1065x, l2_024 354x, l2_034 85x,
l2_003 69x. The first three finals are also 10x to 69x under torch.compile, the other two 2.3x
and 4.4x. l2_096 is one fused kernel that
splits the stride-2 transposed conv into its 8 parity classes and never writes the 127 MB conv
output that eager and torch.compile write and read back.

Nsight Compute was unavailable for 24 of the problems, every other dev problem from
inf_fp8_gemm_tensorwise to l2_034: the host lost the GPU performance-counter permission partway
through the run (`ERR_NVGPUCTRPERM`). Their analysis had no `uncoalesced_access` or `bank_conflict` input.

All 80 changed kernels pass correctness on `ci` and `bench-gpu` again in a fresh worker, every
autotune config included, and none contains the three rejected patterns. compute-sanitizer
memcheck finds no invalid access in any of them on `ci`, every autotune config run. Per-problem
rows are in
`results/pipeline.csv` and `results/pipeline_stages.csv`.

Known limit: float32 inputs. The pipeline checks the declared dtype only. On the upstream
`kernelbench` shapes in float32, which 73 of the changed problems have, 23 kernels pass both as
seed and output, 2 pass only as output, 29 fail as both (seeds that already use `tl.dot`), and 10
run out of memory as both. 9 outputs fail where their seed passes, all declared float16: 7 call
`tl.dot` at Triton's default precision, which rounds float32 operands to TF32, and pass with
`TRITON_F32_DEFAULT=ieee`; l2_067 and l2_085 cast the dot operands to fp16. The seeds compute
these convs with FFMA and are exact in float32. Accumulation is float32 in all 9.

### Stage 6: trials, profiling, Claude Code engine

- Trial tree as in Xe-Forge's `trial_manager.py`:
  - `t0` is the baseline; children record their parent and a sha256 of their source;
  - ranking: correct first, then speedup; a measured result beats an unmeasured one; parity ranks
    at 1.0;
  - `finalize` refuses regressions.
- ncu:
  - profile a single launch (`--kernel-name`, `--launch-skip`);
  - sections: SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, LaunchStats, WarpStateStats,
    SourceCounters;
  - parse the CSV into about 30 lines for the prompt, and diff parent against child.
- nsys for multi-kernel forwards: gaps and launch overhead.
- compute-sanitizer (memcheck, racecheck): run it on candidates that crash or mismatch
  nondeterministically, and on every finalized winner.
- Claude Code workspace generator:
  - contents: `CLAUDE.md`, `config.yaml`, the KB filtered for the device, an `optimize-kernel`
    slash command;
  - rules: only trial files may be written; only `smforge-skill bench` produces timings; the
    oracle is read-only;
  - headless runs through the Claude Agent SDK or `claude -p`, with a turn cap.
- Lessons ledger: a markdown file per kernel recording what was tried and measured. Later runs on
  the same GPU read it.
- A float32 correctness check on the `ci` shapes, so a kernel declared float16 cannot round
  float32 inputs to TF32 or cast them to fp16 (9 of the stage 5 dev kernels do).

Exit: the Claude Code engine optimizes a dev kernel unattended and finalizes a winner. The pipeline
harness re-measures that winner within the A/A band.

Built in v0.6.0, with these differences from the list above:

- The float32 check runs `ci` with the declared float32 inputs, float32 tolerances and a float64
  oracle, and applies only when the kernel a candidate replaces passes it. On the 84 dev problems
  with float32 inputs, 43 seeds pass and 41 fail. All 41 call `tl.dot` without
  `input_precision`, so TF32, and 8 of them also cast to float16 or bfloat16; a hard gate would
  reject them. 10 of the stage 5 finals fail it where their seed passes (on the upstream shapes
  the count was 9). The candidate checks moved to
  `agents/candidate.py`, shared by the stage loop and the trial tree.
- Trial tree (`trials.py`, `smforge-skill trial init|save|status|best|finalize`): `trial save`
  stores the kernel and checks it against `t0` at once, A/B timing included, so every result is
  measured and none can be reported by hand; Xe-Forge also accepts reported numbers. A trial
  whose file changed after its check does not rank. `finalize` takes only a trial faster than
  `t0` (Xe-Forge also finalizes parity) and runs memcheck on it, moving to the next on errors.
- ncu: `smforge-skill profile` runs the six sections on the first launch of each Triton kernel in
  the profiled forward (`--launch-count`, not `--launch-skip`), prints about 30 lines per kernel
  with ncu's whole-kernel estimates, and diffs against `--baseline`. The pipeline's prompts keep
  the counter summary from stage 5; the sections are not added, since each measurement would
  need one more ncu run. Counter access was withdrawn during this stage, so the parser is tested
  on the sample reports shipped with Nsight Compute, not on a live run.
- nsys records the CUDA API calls but no GPU activity on this machine, so the forward timeline
  comes from the PyTorch profiler instead: each forward is profiled alone, and the launch profile
  records the time from the first kernel's start to the last one's end and how much of it no
  kernel ran. `launch_bound` also fires when a multi-kernel forward is idle more than half of that
  span (l2_077's seed: 46.6 of 61.7 us).
- compute-sanitizer: the pipeline runs memcheck on `ci` on the last accepted kernel, and a kernel
  with errors gives way to the stage output before it (`results/pipeline.csv` gains a `memcheck`
  column). A candidate that crashes or mismatches gets memcheck's, else racecheck's, findings in
  its feedback. Nondeterministic mismatches are not told apart from others.
- Finalizing (pipeline, trial tree, Claude Code engine) runs memcheck and then the A/B against
  the seed in three fresh workers; the slowest run is the result and every run is recorded
  (`ab` in the result tables). The pipeline tries accepted kernels from the last stage back. A
  forward of many short kernels times with the host's launch speed, which on this desktop GPU
  drops for seconds at a time: the first exit-run winner on l1_035_groupnorm, 56 launches per
  forward against the seed's 3, measured 1.60x in its trial's single A/B, 1.44x to 1.51x in
  fresh workers, and in about one eager bench in five 1800 to 2650 us against the seed's steady
  1649 us. Under CUDA graphs it was a steady 1056 us. When the three runs spread more than the
  A/A band, the run says so (`ab_spread` in the result tables).
- ncu or compute-sanitizer failing mid-run now stops the run: a failed call reruns the tool
  probe, and if that fails too the run stops with the `--from` that resumes it. Otherwise only
  that kernel goes without. `--from` also resumes after the LLM backend stops.
- Claude Code engine (`claude_ws/`, `smforge claude`): `claude -p /optimize-kernel` with
  `--max-turns`, `--max-budget-usd`, `--permission-mode dontAsk`, the tools Bash, Read, Edit,
  Write, Glob and Grep, and permission rules allowing writes only under `work/` and Bash only for
  `smforge-skill` commands other than `trial init` and `trial finalize`. The rules go through
  `--settings`: Claude Code ignores allow rules in the project settings of a directory nobody has
  trusted. It still runs read-only shell commands (`ls`, `cat`) on its own. Auto memory is off.
  The workspace also holds `stages.md` (each stage's instructions) and `lessons.md`. sm-forge
  finalizes the trial tree itself and checks the winner again in its own worker. The Agent SDK
  is not used.
- Lessons ledger (`lessons.py`): `.smforge/lessons/<gpu>/<problem>.md`, a section per run and a
  line per candidate or trial; the optimizer now starts each reply with one sentence on what it
  changed, which is the line's text. The pipeline puts the last 40 lines in each stage's prompt.

Result (2026-10-31): `smforge claude` with Claude Code 2.1.314, Opus 5.5 at effort medium, up to 8
trials and 80 turns. ncu counter access was off throughout, so `t0` had no counters and no
session called `profile`.

| Problem | Trials | Turns | Cost | Finalized | Trial A/B | Final A/B runs | Recorded |
|---|---|---|---|---|---|---|---|
| l1_035_groupnorm, run 2 | 1 | 19 | $0.55 | t1 | 1.599x | 1.499x (one run; before the three-run finalize) | 1.50x |
| l1_035_groupnorm, run 3 | 1 | 33 | $0.73 | seed kept | 1.58x | 0.612x | 1.00x |
| l1_035_groupnorm, run 4 | 5 | 30 | $0.87 | t3 | 1.590x | 1.579, 1.573, 1.275x | 1.28x |
| l2_057_conv2d_relu_hardswish | 3 | 18 | $0.59 | t1 | 1.294x | 1.294, 1.294, 1.294x | 1.29x |

Run 1 failed at its first API call (Claude Code 2.1.220 does not support Opus 5.5) and was
recorded as a normal finish; a session that fails before its first trial now stops the run.

The exit criterion holds on l2_057: one launch per forward, and the three fresh-worker runs match
the trial's A/B to within 0.05%. The stage 5 pipeline reached 1.25x on it. It does not hold on
l1_035. Every winner there splits the forward into L2-sized slices, 29 to 56 launches against the
seed's 3, and its time follows the host's launch speed: run 4's t3 (56 launches) spread 24%
across its three runs, twelve times the band. The slowest run is recorded and the spread flagged.
Finalize stops at the best trial that passes, so run 4's t4 (29 launches, 1.540x in its trial)
was not tried. Runs 2 and 3 timed variants with `bench` and saved one trial each; after the
prompts described the three-run final check, run 4 measured everything through `trial save`.

The stage 5 finals, re-measured the same way: 78 of the 80 that differ from the seed launch at
most 6 kernels per forward. Of the other two, l2_047_conv3d_mish_tanh (17 launches) measured
3.90, 3.92 and 3.92x against the 3.71x recorded. l1_035_groupnorm (29 launches) recorded 1.54x
and measured 1.53, 0.86 and 1.51x, so the three-run finalize would not keep it.
`results/pipeline.csv` still holds the stage 5 numbers.

### Stage 7: CUDA C++ and CUTLASS tile search

- DSL registry: `triton`, `cuda`.
- CUDA C++ kernels:
  - built as torch extensions with `torch.utils.cpp_extension.load_inline` (ninja,
    `-arch=sm_120`) and cached by source hash;
  - compile errors parsed to file:line for feedback;
  - facts from cuobjdump/nvdisasm and cuda-breakdown;
  - validator for launch bounds, bounds checks, and `cudaGetLastError` after launches.
- KB `cuda/nvidia/`: vectorized loads, shared-memory tiling with padding against bank conflicts,
  `cp.async`, warp-shuffle reductions, `mma.sync` via inline PTX or WMMA, `__launch_bounds__`.
- Tile search:
  - CUTLASS GEMM and Flash Attention templates (Jinja) parameterized by tile shape, stages and
    warp layout;
  - the LLM proposes configs within the smem and register limits, then a
    propose-validate-benchmark loop runs, with cuBLAS as the reference point;
  - first, check which CUTLASS kernels target sm_120, and whether the CuTe Python DSL supports
    sm_120 (it would avoid long C++ builds).

Exit: CUDA C++ seeds go through the same pipeline. Tile search beats CUTLASS's default config on a
set of GEMM shapes.

### Stage 8: evaluation

- `smforge eval --split eval` runs the pipeline over the eval split with a fixed config. It writes
  one JSON file per kernel and a summary.
- Metrics:
  - geomean speedup over eager, over `torch.compile` max-autotune, and over the seed;
  - share of kernels improved; regressions finalized (target 0);
  - correctness on a fresh input draw and on the `ci` variant shapes, which are not used during
    optimization and so catch configs fitted to one shape;
  - kernel-only and forward speedups, reported separately;
  - fraction of roofline; results above roofline flagged as FLOP reduction;
  - LLM calls, tokens, cache-hit share, dollars and wall time per kernel.
- Ablations on the dev split:
  - no KB; KB without arch gating; no facts; no re-analysis; no rollback;
  - T = 1, 3, 5; one pass vs best-of-3;
  - model: Opus 5.5 at several effort levels, Sonnet 5.5 and Haiku 5.5. This tests the paper's
    claim that domain knowledge matters more than model scale.
- Report with roofline plots. Results committed with library versions, driver version and git hash.

Exit: report on the eval split, plus the ablation table.

### Stage 9: host integration (optional)

- `--kernel-repo` for vLLM/SGLang Triton kernels: locate the kernel, copy it as the baseline, build
  a spec from its call site, emit a patch.
- `--external-benchmark` with Xe-Forge's stdout contract (`CORRECTNESS`, `VERDICT`, and `DONE` as
  the last line).
- CI: lint and CPU unit tests on GitHub Actions. Harness and KB tests on a self-hosted GPU runner.
- A second GPU (sm_89 or sm_90) to exercise arch gating.

## 7. Cost

Opus 5.5 costs $4 per MTok input, $20 per MTok output, and $0.20 per MTok for cache reads. Take a
CoVeR call with a 20k-token prompt (15k of it cached) and 8k output tokens including thinking:

| Part | Tokens | Cost |
|---|---|---|
| Uncached input | 5k | $0.020 |
| Cached input | 15k | $0.003 |
| Output | 8k | $0.160 |
| Total per call | | about $0.18 |

At Xe-Forge's typical 10 to 20 calls per kernel, that is $2 to $4 per kernel. One pass over 100
kernels costs about $300, and best-of-k multiplies it. Stage 4 measures the real figure, and the
per-kernel cap is set from it.

## 8. Risks and open questions

- **WSL2.** ncu counters, clock locking, and timing noise from the Windows desktop on the same GPU.
  If stage 0 shows the noise is too large, run evals under native Linux.
- **Triton on sm_120.** Feature support may lag sm_90 and sm_100. The stage 0 probes decide which
  KB entries are enabled.
- **16 GB of memory.** This limits shapes for long-context attention and large MoE. Size spec
  variants to fit.
- **Seed quality.** Seeds set the headline numbers, and seeds with per-call casts inflate them.
  Report kernel-only time and seed provenance.
- **Benchmark gaming.** Covered by the fixed harness, the AST and launch audits, held-out shapes and
  fresh input draws. Some KernelBench Level 2 problems collapse algebraically, which inflates
  geomeans. Report those separately.
- **Cost.** Opus at high effort over 100+ kernels with best-of-k adds up. Use the cap and caching.
- **KB drift.** KB entries can break with new Triton releases. Pin versions, and rerun the KB tests
  on every upgrade.

## 9. References

- Xe-Forge: https://github.com/IntelLabs/Xe-Forge (read at v0.4.0)
- Spoczynski et al., *Xe-Forge: Multi-Stage LLM-Powered Kernel Optimization for Intel GPU*,
  arXiv:2605.26118
- KernelBench: https://github.com/ScalingIntelligence/KernelBench
- KernelAgent / KernelFalcon: https://github.com/meta-pytorch/KernelAgent
- AI-bench (Xe-Forge's timing harness): https://github.com/libxsmm/AI-bench
- cuda-breakdown: `~/projects/optimizer-tool`
