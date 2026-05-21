# PLAN: Rewriting the GEMM Metal Journey in Python + MLX

## Short answer

Yes: rewriting the learning journey around **Python + MLX custom Metal kernels** is a good approach, especially if the goal is to learn GPU optimization, iterate faster, and build richer analysis/visualization tooling.

It is not strictly “better” than C++ for every purpose. C++/Objective-C++ still gives the most direct control over Metal, compilation, command queues, pipeline state, memory allocation, and low-level profiling. But for this repo’s purpose — a learning diary with many kernel variants, benchmarks, plots, and analysis — Python + MLX is likely the better default interface.

The best direction is:

- Use **Python** for orchestration, benchmarking, correctness checks, analysis, plots, and reports.
- Use **MLX custom Metal kernels** for the actual GEMM kernels.
- Keep the current **C++ implementation as a reference/backend archive** until the MLX version reaches parity.

---

## Why move to Python + MLX?

### Pros

1. **Much faster experimentation**
   - Add a new kernel variant without touching CMake, Objective-C++ wrappers, or manual buffer plumbing.
   - Easier to sweep tile sizes, threadgroup sizes, vectorization options, and matrix shapes.

2. **Better benchmark and analysis workflow**
   - Python is ideal for benchmark harnesses, CSV/Parquet output, Pandas analysis, Plotly charts, regression tracking, and roofline modeling.
   - The repo can become both an optimization journey and a reproducible performance-analysis notebook-style project.

3. **Closer to ML workloads**
   - MLX already lives in the Apple Silicon ML ecosystem.
   - It makes it easier to compare against `mlx.matmul`, use realistic model dimensions, and eventually test mixed precision.

4. **Less infrastructure noise**
   - Current C++ Metal code teaches a lot, but some complexity is about API setup rather than GEMM optimization.
   - MLX lets the project focus more on memory hierarchy, tiling, arithmetic intensity, occupancy, and kernel design.

### Cons / risks

1. **Less low-level control than raw Metal**
   - MLX custom kernels abstract away some runtime details.
   - Some advanced Metal features may be harder or impossible to express.

2. **Benchmark overhead must be handled carefully**
   - Python overhead is fine if timing is done correctly, but incorrect synchronization can produce misleading numbers.
   - Need warmups, explicit evaluation/synchronization, stable repetitions, and outlier handling.

3. **Some profiling tools may still require Metal/Xcode knowledge**
   - For deep hardware analysis, Xcode GPU Capture, Metal counters, and Instruments may still be useful.

4. **Performance ceiling may differ**
   - MLX custom kernels can be fast, but raw Metal may still be needed for absolute maximum control.

### Decision

Use MLX as the main path. Keep C++ as a secondary reference. If a kernel requires capabilities MLX does not expose, document that and optionally implement that chapter in C++/Metal.

---

## Project goals

1. Rebuild the GEMM optimization journey in Python using MLX custom Metal kernels.
2. Make every kernel easy to run, benchmark, verify, compare, and plot.
3. Build a stronger benchmark suite covering square, skinny, transformer-like, attention-like, and odd-sized GEMMs.
4. Add roofline analysis to explain *why* kernels perform the way they do.
5. Produce a clean written journey in `README.md` and/or `docs/` with reproducible results.
6. After GEMM, explore ML workloads that need custom backward kernels, such as attention/FlashAttention-style operators.

---

## Proposed repository structure

```text
.
├── gemm_metal/
│   ├── __init__.py
│   ├── kernels/
│   │   ├── naive.py
│   │   ├── tiled.py
│   │   ├── register_tiled.py
│   │   ├── simdgroup.py
│   │   └── utils.py
│   ├── benchmark/
│   │   ├── runner.py
│   │   ├── cases.py
│   │   ├── metrics.py
│   │   ├── roofline.py
│   │   └── report.py
│   ├── reference.py
│   └── config.py
├── scripts/
│   ├── bench.py
│   ├── plot_roofline.py
│   ├── plot_kernel_comparison.py
│   └── validate.py
├── notebooks/
│   └── analysis.ipynb
├── outputs/
│   ├── raw/
│   ├── plots/
│   └── reports/
├── cpp_legacy/              # optional: move current C++ version here later
├── README.md
├── PLAN.md
└── pyproject.toml
```

---

## Implementation phases

## Phase 0: Baseline migration setup

- Add MLX dependency to `pyproject.toml`.
- Create Python package layout.
- Add a small MLX smoke test:
  - Allocate matrices.
  - Run `mx.matmul`.
  - Verify against NumPy for small sizes.
- Add a simple benchmark timer utility with:
  - Warmup iterations.
  - Repeated measured iterations.
  - Explicit synchronization/evaluation.
  - Median, mean, min, max, std.

Deliverables:

- `gemm_metal/reference.py`
- `gemm_metal/benchmark/runner.py`
- `scripts/bench.py`

---

## Phase 1: Naive MLX custom Metal GEMM

Implement the first custom Metal kernel through MLX:

- One output element per thread.
- FP32 only.
- Row-major layout.
- Support arbitrary `M`, `N`, `K`.
- Validate against `mx.matmul` and NumPy on small shapes.

Metrics to record:

- Runtime in ms.
- GFLOP/s.
- Effective memory bandwidth estimate.
- Arithmetic intensity estimate.
- Max absolute error.
- Mean absolute error.

Deliverables:

- `gemm_metal/kernels/naive.py`
- benchmark CSV/Parquet output for naive kernel
- README chapter: naive baseline

---

## Phase 2: Benchmark suite upgrade

Create a proper benchmark matrix catalog.

### Shape families

1. **Square powers of two**
   - `(512, 512, 512)` through `(4096, 4096, 4096)` if memory allows.

2. **Transformer FFN-like**
   - Examples:
     - `(M, K) @ (K, N)` where `M = batch * seq_len`
     - Llama-style sizes: `K = hidden_dim`, `N = intermediate_dim`

3. **Attention-like**
   - QK and AV shapes.
   - Smaller `K`, larger sequence dimensions.

4. **Skinny/tall matrices**
   - Stress memory traffic and poor reuse.

5. **Odd/non-ideal sizes**
   - Sizes not divisible by tile sizes.
   - Useful for boundary handling and real-world robustness.

6. **Small matrices**
   - Important because launch overhead and occupancy dominate.

### Benchmark output schema

Each benchmark row should include:

- timestamp
- git commit hash
- machine/chip info if available
- Python version
- MLX version
- kernel name
- dtype
- M, N, K
- shape family
- warmup iterations
- measured iterations
- mean/median/min/max runtime
- std runtime
- GFLOP/s
- bytes read/write estimate
- arithmetic intensity
- estimated bandwidth
- correctness error

Deliverables:

- `gemm_metal/benchmark/cases.py`
- `gemm_metal/benchmark/metrics.py`
- `outputs/raw/*.csv` or `*.parquet`

---

## Phase 3: Roofline model

Add roofline analysis to understand whether each kernel is memory-bound or compute-bound.

### Roofline basics

For GEMM `C = A @ B` with shape `(M, K) @ (K, N)`:

- FLOPs: `2 * M * N * K`
- Minimal memory traffic estimate for FP32:
  - read A: `M * K * 4` bytes
  - read B: `K * N * 4` bytes
  - write C: `M * N * 4` bytes
- Arithmetic intensity:
  - `FLOPs / bytes`

Performance ceiling:

```text
achievable_GFLOPS = min(peak_compute_GFLOPS, arithmetic_intensity * peak_bandwidth_GBps)
```

### Need to estimate machine constants

For the M2 8-core GPU:

- FP32 peak: current README uses about `2840 GFLOP/s`.
- Memory bandwidth: use measured bandwidth if possible, otherwise start with published unified memory bandwidth.

Better: add microbenchmarks for:

- copy bandwidth
- vector add bandwidth
- MLX `matmul` peak-ish compute reference

Deliverables:

- `gemm_metal/benchmark/roofline.py`
- `scripts/plot_roofline.py`
- `outputs/plots/roofline.html`
- README section explaining roofline results

---

## Phase 4: Tiled kernels

Implement tiled GEMM variants using threadgroup memory.

Kernel variants:

1. `tile_8x8`
2. `tile_16x16`
3. `tile_32x32` if feasible
4. rectangular tiles such as `16x8`, `8x16`, `32x8`

Study:

- tile size vs occupancy
- threadgroup memory usage
- boundary overhead
- memory coalescing
- effect on arithmetic intensity and measured bandwidth

Deliverables:

- `gemm_metal/kernels/tiled.py`
- comparison plots against naive kernel
- README chapter: tiling

---

## Phase 5: Register tiling / multiple outputs per thread

Implement kernels where each thread computes multiple C elements.

Variants:

- 1x2 outputs per thread
- 2x2 outputs per thread
- 4x1 outputs per thread
- 4x4 if register pressure allows

Study:

- register reuse
- instruction-level parallelism
- occupancy loss from register pressure
- best shape-dependent tile choices

Deliverables:

- `gemm_metal/kernels/register_tiled.py`
- benchmark comparison by shape family
- README chapter: register tiling

---

## Phase 6: SIMD-group / matrix primitives

Port the current simdgroup idea to MLX custom Metal kernels if MLX supports the needed Metal code path.

Study:

- simdgroup matrix multiply primitives
- FP32 behavior on Apple GPU
- whether this maps to specialized matrix hardware
- constraints on tile sizes and alignment

If MLX blocks some required feature, document it clearly and keep this as a raw Metal/C++ chapter.

Deliverables:

- `gemm_metal/kernels/simdgroup.py` if possible
- fallback note if raw Metal is required
- README chapter: simdgroup kernels

---

## Phase 7: Better reporting

Create a reproducible report pipeline.

Reports should include:

- best kernel per shape
- GFLOP/s by kernel and shape family
- percentage of theoretical peak
- roofline chart
- speedup over naive
- speedup over previous kernel
- comparison to `mlx.matmul`
- correctness summary

Deliverables:

- `scripts/bench.py` for running the full fixed benchmark suite outside the package CLI
- `scripts/report.py` for generating `outputs/reports/latest.md`
- `scripts/plot_kernel_comparison.py` for simple kernel comparison HTML
- `scripts/plot_roofline.py` for simple roofline HTML
- `outputs/reports/latest.md`
- HTML charts in `outputs/plots/`

---

## Phase 8: Custom autograd and attention experiments

MLX custom Metal kernels can participate in autograd when paired with a manually written VJP/backward rule. This means the project can go beyond forward-only GEMM kernels.

Possible experiments:

1. Write a tiny custom op with a handwritten forward kernel and handwritten VJP.
2. Add a reduction/softmax forward kernel.
3. Add a softmax backward kernel.
4. Build a toy scaled-dot-product attention forward path.
5. Attempt a FlashAttention-style forward kernel using block tiling and online softmax.
6. Attempt a FlashAttention-style backward path using manually written VJP kernels.

Important notes:

- MLX will not automatically differentiate through arbitrary Metal source.
- Backward support requires explicit VJP/JVP registration and one or more custom kernels.
- Atomics and `atomic_outputs=True` are useful for scatter-style gradients, but high-performance attention backward should avoid excessive atomics where possible.
- Some sophisticated kernels may require multiple custom kernels rather than one giant kernel.

Deliverables:

- `src/gemm_metal/attention/` or `src/gemm_metal/ops/` experiments
- correctness tests against pure MLX reference implementations
- forward and backward benchmark reports

---

## Phase 9: Documentation rewrite

Rewrite the README as a polished learning journey:

1. Introduction: why GEMM, why Apple Metal, why MLX.
2. Benchmarking methodology.
3. Naive kernel.
4. Tiling.
5. Register tiling.
6. SIMD-group/matrix primitives.
7. Roofline model.
8. Lessons learned.
9. How to reproduce results.

Keep the current README content as historical reference while migrating.

---

## Benchmarking rules

To avoid misleading results:

- Always warm up kernels before measuring.
- Synchronize/evaluate before and after timing.
- Use median runtime as the primary number.
- Report variance, not just best time.
- Separate kernel time from setup/allocation time.
- Reuse input/output buffers where possible.
- Validate correctness separately from performance timing.
- Compare against `mlx.matmul` as a strong baseline.
- Keep raw benchmark outputs committed only if small; otherwise commit summary reports and plots.

---

## Success criteria

The rewrite is successful if:

- Every kernel can be run from Python with one command.
- Benchmarks are reproducible and saved with rich metadata.
- Plots clearly show performance progression.
- Roofline analysis explains bottlenecks better than GFLOP/s alone.
- The Python/MLX version reaches similar or better educational value than the C++ version.
- The repo becomes easier to extend with new kernels and new analysis.

---

## Initial task checklist

- [ ] Add `mlx` dependency.
- [ ] Create Python package structure.
- [ ] Implement benchmark runner.
- [ ] Implement shape catalog.
- [ ] Implement naive MLX custom kernel.
- [ ] Add correctness tests.
- [ ] Add CSV/Parquet benchmark output.
- [ ] Add roofline metric calculations.
- [ ] Add first roofline plot.
- [ ] Port tiled kernel.
- [ ] Port register-tiled kernel.
- [ ] Investigate simdgroup support in MLX custom kernels.
- [ ] Rewrite README around the new journey.

---

## Recommendation

Proceed with the Python + MLX rewrite. It fits the learning and analysis goals better than continuing only in C++. Do not delete the C++ code immediately; keep it as a reference until the MLX path is mature.
