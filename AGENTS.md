# AGENTS.md

Guidance for coding agents working on this repository.

## Project purpose

This repo is a learning journey for writing and benchmarking GEMM kernels on Apple Silicon using **Python + MLX custom Metal kernels**.

The important goal is educational clarity, not just peak performance. Code should make it easy to:

- add one kernel variant at a time,
- verify correctness,
- benchmark reproducibly,
- analyze results with metrics such as GFLOP/s, bandwidth, arithmetic intensity, and roofline position,
- document what changed and why.

## Very important: kernel ownership

The Metal kernel implementations are intended to be written by the repo owner by hand.

Agents should **not** silently implement or replace custom Metal GEMM kernels unless explicitly asked.

Allowed kernel-related work:

- create scaffolding files,
- add TODO placeholders,
- wire benchmark/validation interfaces,
- add very thin Python helpers for MLX custom kernel body strings,
- improve docstrings and comments,
- add correctness checks,
- add benchmark plumbing,
- explain what a kernel should do,
- review a user-written kernel when asked.

Avoid unless explicitly requested:

- writing full Metal kernel source strings,
- optimizing kernel math loops,
- changing tile sizes inside a kernel implementation,
- replacing a user-written kernel with generated code,
- hiding logic inside helper functions that effectively implement the kernel.

If a kernel file contains a `HANDWRITTEN KERNEL AREA` marker, do not edit inside that area without direct permission.

MLX custom Metal kernels in this repo should live as Python raw strings in `src/gemm_metal/kernels/*.py`. MLX generates the Metal function signature, so these strings should contain only the kernel body. The repo owner should also control launch details inside each kernel file: input names, output names, template values, grid size, threadgroup size, output shapes, and dtypes. Agents may edit generic Python wrappers/helpers but must not fill in kernel body strings or choose kernel launch parameters unless explicitly asked.

## Code style

- Keep Python simple and readable.
- Prefer standard library dependencies unless a dependency is clearly useful.
- Type annotate public functions.
- Keep benchmark code deterministic and explicit.
- Keep generated outputs under `outputs/` and do not require them for normal imports.

## Benchmarking rules

- Do not include allocation/setup time in kernel timings unless the benchmark explicitly says so.
- Always warm up before measuring.
- Always force MLX evaluation/synchronization around timed regions using `mx.eval(...)` on outputs.
- Use median time as the primary result.
- Record enough metadata to reproduce the run.
- Validate correctness separately from speed measurements.

## Suggested commands

```bash
uv run gemm-metal --help
uv run gemm-metal list-kernels
uv run gemm-metal validate --kernel mlx --M 128 --N 128 --K 128
uv run gemm-metal bench --kernel mlx
```

## Repository layout

```text
src/gemm_metal/
  benchmark/      Fixed benchmark case catalog, metrics, runner, roofline helpers
  kernels/        Kernel registry, launch helpers, and user-written kernel body strings
  cli.py          Command-line interface
  reference.py    Reference implementations
  utils.py        Small shared utilities
outputs/          Benchmark outputs, plots, reports
```
