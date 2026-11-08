# Security policy

## Supported versions

Security fixes go into the latest release only.

| Version | Supported |
|---|---|
| 1.0.x | yes |
| 0.x | no |

## Reporting a vulnerability

Please don't report security problems in a public issue. Report them privately from the
repository's **Security** tab: **Report a vulnerability**. Only you and the maintainer see the
report.

Include:

- the sm-forge version or commit
- what you ran, and how to reproduce the problem
- what an attacker could do with it

I will reply as soon as I can, work on a fix with you, and credit you in the advisory unless you
prefer not to be named. Please keep the details private until a fix is released.

## sm-forge runs code an LLM writes

This is how sm-forge works, not a vulnerability, but it decides where you should run it.

- Every candidate kernel is a Python module, and CUDA C++ kernels are compiled with nvcc and
  loaded. sm-forge imports and runs them in a worker subprocess. The worker keeps a GPU fault
  from taking down the main process. It is not a sandbox: a kernel file can do anything your
  user account can.
- The static checks and the launch audit decide whether a kernel counts as correct and fast.
  They are not a security boundary.
- The [Claude Code engine](docs/claude-code-engine.md) restricts its session to reading files,
  editing under `work/`, and running `smforge-skill`. `smforge-skill` then runs the kernels the
  session wrote, so the same applies.

Run sm-forge on a machine or in a container where that is acceptable. The kernels run as your
user, so they can read `.env` with its API key, and any other file you can. Keep other
credentials out of that environment.

## In scope

- A way for an LLM reply or a problem's files to run code on the host without going through the
  worker, for example during prompt building, result parsing or report generation
- A Claude Code session writing outside its workspace's `work/` or running commands other than
  `smforge-skill`
- The API key, or other secrets, ending up in prompts, transcripts, results or anything else
  sm-forge writes or commits

## Out of scope

- Vulnerabilities in PyTorch, Triton, CUDA, CUTLASS or Claude Code. Report those to their
  projects.
- A kernel that passes the checks while being wrong or using a library. Open a normal issue for
  that.
