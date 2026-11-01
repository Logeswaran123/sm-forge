# Optimize one Triton kernel: $problem

`kernel.py` is a correct Triton kernel for the PyTorch reference in `problem/reference.py`.
Make it faster on $device, Triton $triton, PyTorch $torch, without changing what it computes.

## Rules

- Write files only under `work/`. `problem/`, `kernel.py`, `trials/`, `kb.md`, `stages.md`,
  `lessons.md` and this file are read-only.
- Timings come only from `smforge-skill`. Do not write timing code, benchmark scripts or test
  harnesses of your own, and do not run Python directly.
- The reference and the spec are the oracle. Do not change them, copy their outputs, or work
  around a check.
- A kernel counts only once `smforge-skill trial save` has stored and checked it. When you
  stop, the harness takes the best saved trial; nothing you copy elsewhere is used.

## Problem

$task

## Measurements of `kernel.py` (trial `t0`)

$measurements

Issues these measurements establish:

$issues

## How a trial is checked

`smforge-skill trial save` runs these in order against `t0` and reports the first failure:

1. Static checks and an audit of `forward`: all computation runs in `@triton.jit` kernels;
   `forward` and its helpers may only allocate tensors and change views, layouts and dtypes.
   `forward` keeps no state between calls (no assignment to `self`), and has no host
   synchronization.
2. The interface: `Model.__init__` keeps its arguments and creates the same parameters and
   buffers; `get_inputs` and `get_init_inputs` stay as they are.
3. Correctness against the reference on `ci` (odd sizes) and `bench-gpu`, every autotune config
   included$float32_check.
4. The launch profile: no cuBLAS, cuDNN, ATen compute or other non-Triton kernel that `t0` does
   not launch.
5. Interleaved timing on `bench-gpu` against `t0`. A trial is faster only by more than
   $min_effect; within that it is parity.

At the end the best trial that is faster than `t0` also has to pass compute-sanitizer memcheck
on `ci`.

## Commands

```bash
smforge-skill validate work/k.py --spec problem/spec.yaml
smforge-skill trial save work/k.py --parent t0 --strategy "what this trial changes"
smforge-skill trial status
smforge-skill trial best
smforge-skill check work/k.py --reference problem/reference.py --spec problem/spec.yaml --variant ci
smforge-skill bench work/k.py --baseline kernel.py --spec problem/spec.yaml
smforge-skill facts work/k.py --spec problem/spec.yaml
smforge-skill sanitize work/k.py --spec problem/spec.yaml --variant ci
```

`trial save` takes a minute or two: it runs every check above. `--parent` names the trial the
new one starts from. Give each trial its own file under `work/`.

## Reading

- `kb.md`: constraints every kernel must respect, and optimization patterns by stage, each with an
  example measured on this GPU's architecture.
- `stages.md`: what each optimization stage changes.
- `lessons.md`: what earlier runs on this problem and GPU tried and measured, if any.
