# gemm-metal

Small MLX custom Metal kernel experiments on Apple Silicon.

The project mirrors the cuda-lab structure:

```text
src/gemm_metal/
  main.py
  harness/
    bench.py
    check.py
  kernels/
    utils.py
    02_matmul/
      bench.py
      test.py
      reference.py
      naive.py
      tiled.py
```

`harness/` contains reusable benchmark and correctness helpers. Each folder under
`kernels/` owns its reference implementation, tests, benchmark script, and Metal
kernel wrappers.

## Setup

```bash
uv sync
uv run gemm-metal list
```

## Matrix Multiplication

Run correctness tests:

```bash
uv run gemm-metal test 02_matmul
```

Run benchmarks:

```bash
uv run gemm-metal bench 02_matmul --kernel naive
uv run gemm-metal bench 02_matmul --kernel tiled_16
uv run gemm-metal bench 02_matmul --kernel tiled_32
```

Every matmul implementation takes explicit `M`, `N`, and `K`:

```python
C = matmul_naive(A, B, M, N, K)
```

### Metal Launch Indexing

The naive kernel uses CUDA-like names in the Metal body, but MLX's Metal launch
contract is not the same as CUDA's launch syntax.

CUDA usually launches with:

```text
grid  = number of blocks
block = threads per block
```

MLX `mx.fast.metal_kernel` launches with:

```text
grid        = total thread grid
threadgroup = threads per threadgroup
```

So this CUDA-style launch:

```text
grid=(blocks_x, blocks_y)
block=(16, 16)
```

is represented in this project as:

```text
threadgroup=(256, 1, 1)
grid=(blocks_x * 256, blocks_y, 1)
```

Inside the kernel:

```text
threadgroup_position_in_grid.x -> logical block_x
threadgroup_position_in_grid.y -> logical block_y
thread_position_in_threadgroup.x -> flat local thread id
```

The flat local thread id is converted back to a logical 16x16 tile index:

```text
thread_x = thread_id % 16
thread_y = thread_id / 16
```

Your SIMD-group comparison is still the right mental model:

```text
Metal threadgroup ~= CUDA block
Metal SIMD-group  ~= CUDA warp
```

The difference above is about the MLX Python launch API: its `grid` parameter is
the full Metal thread grid, not CUDA's number of blocks.

Shared validation lives in `gemm_metal.kernels.utils` and checks:

- `M`, `N`, and `K` are positive integers.
- `A` and `B` are 2D MLX arrays.
- `A` has shape `(M, K)`.
- `B` has shape `(K, N)`.
- `A` and `B` use the same supported dtype.

The current kernels support `mx.float32`.
