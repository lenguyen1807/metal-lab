# A journey to ~2.84 TFLOPs on my M2 MacBook

This is a diary of my journey to write a fast single-precision floating point (FP32) matrix multiplication (SGEMM) kernel on my Apple M2 laptop.

Since I do not have an NVIDIA card lying around, I am using Apple's **Metal** ecosystem instead of CUDA. The core ideas are the same: make GEMM as fast as possible by understanding memory movement, tiling, reuse, occupancy, and the hardware execution model.

The new version of this repo uses **Python + MLX custom Metal kernels**. Python handles the boring-but-important parts: correctness checks, benchmark orchestration, CSV output, roofline calculations, and reports. The kernel bodies themselves are still handwritten Metal code, but MLX generates the function signature and launches the kernel for us.

The theoretical FP32 peak of an 8-core M2 GPU is about **2.84 TFLOPs** (2840 GFLOP/s) [^1] [^2]. Can we get close? Let's find out.

This repo is for anyone who wants to learn GPU optimization but does not have an NVIDIA GPU. It is also an excuse to make benchmarking and performance analysis less hand-wavy.

## How fast are we so far?

No custom kernel numbers yet. This Python/MLX rewrite is starting from a clean slate.

Current baseline:

- `mlx`: calls `mlx.matmul`; used as the correctness and performance reference.

Planned handwritten kernels:

- `naive`: one output element per thread.
- `tiled`: use Metal `threadgroup` memory to improve input reuse.
- `register_tiled`: compute multiple output elements per thread.
- `simdgroup`: use SIMD-group / matrix primitives where possible.

## Benchmark plan

We benchmark many matrix sizes (`M x N x K`) instead of only one square matrix. Different shapes stress different parts of the GPU.

- **Powers of 2 / square**: `M=N=K`, useful as a clean baseline.
- **FFN layers**: transformer feed-forward shapes, usually large and compute-heavy.
- **Attention layers**: sequence-length GEMMs with relatively small head dimensions.
- **Skinny matrices**: small `K`, often more memory-bandwidth sensitive.
- **Odd / non-ideal sizes**: dimensions not divisible by tile sizes, useful for testing boundary handling.

The benchmark harness records runtime, GFLOP/s, estimated bandwidth, arithmetic intensity, and correctness error.

## Roofline model

For a GEMM `C = A @ B` with dimensions `(M, K) @ (K, N)`:

- multiply-add operations: `M * N * K`
- FLOPs, counting one FMA as two operations: `2 * M * N * K`
- ideal input elements: `(M * K) + (N * K)`
- output elements: `M * N`
- ideal total elements moved: `(M * K) + (N * K) + (M * N)`

Reference-style arithmetic intensity [^3]:

```text
ops per input element  = (M * N * K) / ((M * K) + (N * K))
                       = (M * N) / (M + N)

ops per output element = (M * N * K) / (M * N)
                       = K

ops per total element  = (M * N * K) / ((M * K) + (N * K) + (M * N))
```

For roofline plots, this repo also reports FLOP/byte using the ideal FP32 traffic estimate. This assumes perfect input reuse: each input element is loaded from device memory once and reused from cache/threadgroup memory after that. Real kernels only approach this with careful tiling.

## Get it running

> [!IMPORTANT]
> You'll need **a Mac with an M-series chip**.

This project uses [`uv`](https://docs.astral.sh/uv/) and MLX.

```bash
uv sync
```

## Run the benchmark

List available kernels:

```bash
uv run gemm-metal list-kernels
```

Validate a kernel against `mlx.matmul`:

```bash
uv run gemm-metal validate --kernel mlx --M 128 --N 128 --K 128
```

Run the benchmark suite:

```bash
uv run gemm-metal bench --kernel mlx
```

Benchmark a custom kernel. The production `mlx.matmul` baseline is always included in the same output file for comparison:

```bash
uv run gemm-metal bench --kernel naive
```

The script wrapper does the same thing:

```bash
uv run python scripts/bench.py --kernel mlx
```

Generate a Markdown report from benchmark CSV files:

```bash
uv run python scripts/report.py
```

Generate simple HTML plots:

```bash
uv run python scripts/plot_kernel_comparison.py
uv run python scripts/plot_roofline.py
```

Benchmark outputs live under `outputs/`.

## The optimization checklist

This project is structured so you can follow the optimization journey step by step. Each new kernel should be a separate chapter and a separate implementation.

- [x] **Chapter 0: The Python + MLX setup** Build a clean benchmark harness with correctness checks and reproducible outputs (Thanks Codex).
- [ ] **Chapter 1: Naive GEMM** One output element per thread. The humble beginning.
- [ ] **Chapter 2: Tiling** Use `threadgroup` memory, the Metal equivalent of CUDA shared memory, to reduce device-memory traffic.
- [ ] **Chapter 3: More work per thread / register tiling** Compute small output blocks per thread to increase reuse and hide latency.
- [ ] **Chapter 4: SIMD-group matrix primitives** Use Metal SIMD-group features where MLX custom kernels allow it.
- [ ] **Chapter 5: Better analysis** Roofline plots, shape-family comparisons, and performance reports.
- [ ] **Chapter 6: Beyond GEMM** Try custom backward kernels, attention, and maybe a FlashAttention-style kernel.

## Kernel philosophy

MLX custom Metal kernels are defined from a Metal **function body string**. MLX generates the function signature.

The handwritten kernel body and launch choices are the main learning artifacts in this repo. Each kernel file should own its own `mx.fast.metal_kernel(...)` setup: input names, output names, templates, grid size, threadgroup size, output shapes, and dtypes. Tooling and agents may help with scaffolding, benchmark code, validation, and reports, but the actual kernel bodies and launch parameters should be written by hand.

See `AGENTS.md` for detailed contributor/agent instructions.

## Resources

- [MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
- [siboehm's CUDA Matrix Optimization](https://siboehm.com/articles/22/CUDA-MMM)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [metal_performance_testing by bkvogel](https://github.com/bkvogel/metal_performance_testing)
- [metal_flash_attention by philipturner](https://github.com/philipturner/metal-flash-attention/tree/main)
- [Outperforming cuBLAS on H100: a Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)

[^1]: https://www.cpu-monkey.com/en/cpu-apple_m2_8_gpu
[^2]: https://github.com/philipturner/metal-benchmarks
[^3]: https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf