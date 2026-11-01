# Profiling tools

Nsight Compute (`ncu`), Nsight Systems (`nsys`) and `compute-sanitizer` come with the CUDA
toolkit. sm-forge finds them on `PATH`, under `CUDA_HOME`, or in `/usr/local/cuda*`. All three
are optional. Check which work here:

```bash
smforge device --tools
```

| Tool | Used for | Without it |
|---|---|---|
| ncu | counters for the analyzer (coalescing, bank conflicts), `smforge-skill profile` | the issues that need counters go undetected |
| compute-sanitizer | memcheck on final kernels, findings for crashing candidates, `smforge-skill sanitize` | finals are not memchecked |
| nsys | not used for timing: under WSL2 it records no GPU activity, so launch timelines come from the PyTorch profiler | nothing lost |

## ncu: performance counter access

ncu fails with `ERR_NVGPUCTRPERM` until counters are open to all users
([NVIDIA's page](https://developer.nvidia.com/ERR_NVGPUCTRPERM)).

- **Linux:** load the driver with `NVreg_RestrictProfilingToAdminUsers=0`.
- **WSL2:** the setting is on Windows. In NVIDIA Control Panel, Desktop > Enable Developer
  Settings, then Developer > Manage GPU Performance Counters > Allow access to all users. Then run
  `wsl --shutdown` and start WSL again.

> [!WARNING]
> Counter access can drop in the middle of a long run. A pipeline run that sees ncu stop working
> stops and prints the `--from` that resumes it. Check `smforge device --tools` before a long run.

## compute-sanitizer under WSL2

It needs the Windows debugger interface: run `EnableDebuggerInterface.bat` from the Windows CUDA
toolkit as administrator, then reboot.
