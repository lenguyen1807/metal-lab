# metal-kernel-notes

Small MLX custom Metal kernel experiments on Apple Silicon.

The project is organized like a compact CUDA learning repo: each operator keeps
its handwritten kernels in `ops/`, and each operator has a matching benchmark
spec in `bench/`.

## Layout

```text
src/gemm_metal/
  main.py
  ops/
    vadd.py
    gemm.py
  bench/
    common.py
    vadd.py
    gemm.py
scripts/
  bench.py
outputs/
  raw/
```

## Setup

```bash
uv sync
uv run gemm-metal list
```

## Kernels

```text
vadd
  mlx       MLX elementwise add baseline
  metal     one Metal thread per element

gemm
  mlx       MLX matmul baseline
  simt      naive CUDA-style GEMM, one thread per C element
  tiled16   threadgroup-memory tiled GEMM, 16x16 output tile
  tiled32   threadgroup-memory tiled GEMM, 32x32 output tile
  simdgroup SIMD-group matrix GEMM, threadgroup computes 32x16 output tile
```

The GEMM kernels intentionally use CUDA-like naming in the Metal body:

```text
threadgroup_position_in_grid      -> block position
thread_position_in_threadgroup    -> thread position
```

## GEMM Kernel Notes

`simt`:

```text
C[row, col] = sum_k A[row, k] * B[k, col]
```

`tiled16` and `tiled32`:

```text
load A/B tiles into threadgroup memory
synchronize
accumulate one C element per thread
```

`simdgroup`:

```text
256 threads per threadgroup
8 SIMD-groups per threadgroup
each SIMD-group computes one 8x8 C tile
the threadgroup computes a 32x16 C tile
```

The current SIMD-group kernel requires `M`, `N`, and `K` to be divisible by 8.
Unsupported benchmark cases are skipped.

## Benchmark Commands

Validate:

```bash
uv run gemm-metal validate vadd --kernel metal
uv run gemm-metal validate gemm --kernel simt
uv run gemm-metal validate gemm --kernel tiled16
uv run gemm-metal validate gemm --kernel tiled32
uv run gemm-metal validate gemm --kernel simdgroup
```

Benchmark square GEMM:

```bash
uv run gemm-metal bench gemm --kernel simt --suite square
uv run gemm-metal bench gemm --kernel tiled16 --suite square
uv run gemm-metal bench gemm --kernel tiled32 --suite square
uv run gemm-metal bench gemm --kernel simdgroup --suite square
```

Benchmark broader or heavier suites:

```bash
uv run gemm-metal bench gemm --kernel simt --suite full
uv run gemm-metal bench gemm --kernel simdgroup --suite full
uv run gemm-metal bench gemm --kernel simt --suite stress --iterations 3
uv run gemm-metal bench gemm --kernel mlx --suite extreme --iterations 3
```

The script wrapper exposes the same command:

```bash
uv run python scripts/bench.py bench gemm --kernel tiled16 --suite square
```

When a custom kernel is benchmarked, the MLX baseline is included first in the
same output.

## Suites

```text
square   square GEMMs from 256 to 2048, for headline comparisons
full     square, FFN-like, attention-like, skinny/wide/tall, and odd cases
stress   intentionally large GEMM cases
extreme  very large model-like cases; best for MLX or optimized kernels first
```

`stress` and `extreme` can be slow for naive kernels. Start with low iteration
counts.

## Metrics

Each CSV row records:

```text
timestamp
op, kernel, case, family, shape
warmups, iterations
mean/median/min/max/std runtime
GFLOP/s
estimated bandwidth
arithmetic intensity
max/mean absolute error
Python version
MLX version
```

Timing excludes input allocation/setup. The runner warms up first, times only
kernel execution, and forces MLX evaluation with `mx.eval(...)`.

## Reporting

Do not average GFLOP/s across unrelated shapes. A single average hides whether a
kernel is good at square GEMM, skinny GEMM, attention-like GEMM, or odd boundary
cases.

Prefer:

```text
case | shape | MLX GFLOP/s | custom GFLOP/s | custom / MLX | max error
```

Then group results by family:

```text
square
FFN-like
attention-like
skinny/wide/tall
odd sizes
```

Use the large `square` suite for headline numbers, and use `full` to explain
where a kernel works or breaks down. A median custom/MLX ratio within one family
is useful as a secondary summary, but it should not replace the per-shape table.

## Roadmap

1. Vector add.
2. GEMM SIMT naive.
3. GEMM tiled.
4. GEMM SIMD-group.
5. Reduction sum.
6. Softmax.
7. Attention.
