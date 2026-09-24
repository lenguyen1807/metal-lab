# M5 GEMM experiments

This is a worklog for packed, row-major, non-transposed FP32 GEMM (`C = A @ B`),
not a claim about every matrix shape or precision. Each shader variant stays in
the repository so a slower result remains inspectable. Use `gemm list` to see
exact function names and `--function` to repeat one experiment.

## Hardware and measurement

- Machine used here: Apple M5 GPU, macOS 26.7, Xcode 27, Homebrew MLX 0.32.1.
- Main shape: `M = N = K = 4096`, approximately 137.44 billion floating-point
  operations by the usual `2MNK` convention.
- The reported number is the median of five host `steady_clock` timings of a
  **completed call**, after one correctness run of each supported kernel.
  MPS and custom calls include command encoding, submission, and waiting; MLX
  calls include graph construction, evaluation, and synchronization. Inputs are
  populated before timing. MLX allocates its output during the call, whereas
  MPS and custom kernels reuse output buffers.
- All rows are checked against the same MPS result using
  `abs(result - MPS) <= 1e-3 + 1e-3 * abs(MPS)`. This is an agreement test,
  not a full accuracy study. The benchmark cannot isolate GPU execution time.
- Absolute times changed across runs, so compare each custom row only with the
  MPS and MLX baselines in its own table. Thermal state, GPU frequency, and
  other work on the machine may affect results.

## TensorOps tile and K-loop sweep

These rows came from **one** `4096³` run with five timing iterations. The
`tensorops` functions let `matmul2d` handle the K dimension dynamically.
`tensorops_sync` slices K explicitly, keeps a cooperative tensor accumulator,
and synchronizes four SIMD groups between chunks. Their descriptor explicitly
sets `relaxed_precision = false`.

| Function | Median ms | GFLOP/s | MPS time / own time | MLX time / own time |
| --- | ---: | ---: | ---: | ---: |
| MPS | 41.0721 | 3346.3 | 1.00× | 0.99× |
| MLX | 40.6651 | 3379.8 | 1.01× | 1.00× |
| `tensorops_64x64` | 45.0341 | 3051.9 | 0.91× | 0.90× |
| `tensorops_64x128` | 42.9849 | 3197.4 | 0.96× | 0.95× |
| `tensorops_128x64` | 47.4999 | 2893.5 | 0.86× | 0.86× |
| `tensorops_128x128` | 120.8816 | 1137.0 | 0.34× | 0.34× |
| `tensorops_64x64_morton` | 49.5674 | 2772.8 | 0.83× | 0.82× |
| `tensorops_32x32_sg1` | 64.0579 | 2145.5 | 0.64× | 0.63× |
| `tensorops_32x32_sg1_morton` | 71.2417 | 1929.2 | 0.58× | 0.57× |
| `tensorops_32x64_sg1` | 43.4235 | 3165.1 | 0.95× | 0.94× |
| `tensorops_64x32_sg1` | 47.5024 | 2893.3 | 0.86× | 0.86× |
| `tensorops_sync_k128` | 36.9892 | 3715.7 | 1.11× | 1.10× |
| **`tensorops_sync_k256`** | **36.7900** | **3735.8** | **1.12×** | **1.11×** |
| `tensorops_sync_k512` | 37.4169 | 3673.2 | 1.10× | 1.09× |
| `tensorops_sync_k1024` | 37.9778 | 3618.9 | 1.08× | 1.07× |
| `tensorops_sync_64x128_k512` | 37.7635 | 3639.5 | 1.09× | 1.08× |
| `tensorops_sync_32x64_k512` | 37.8135 | 3634.7 | 1.09× | 1.08× |
| `tensorops_sync_64x128_k256` | 37.3580 | 3679.0 | 1.10× | 1.09× |
| `tensorops_sync_k256_morton` | 37.3855 | 3676.3 | 1.10× | 1.09× |

The synchronized K variants clustered closely. The 256-wide chunk was fastest
in this run, while the Morton variant did not improve this square workload.
The large 128×128 output tile was much slower. A separate ten-iteration run
of only `tensorops_sync_k256` gave 39.9587 ms versus MPS 44.0723 ms and MLX
43.0978 ms, preserving the roughly 1.1× advantage under that run's conditions.

## SIMD-group matrix sweep

This was a **different** five-iteration `4096³` run. All three functions use
the older 8×8 `simdgroup_matrix` primitive; the larger versions reuse matrix
fragments in registers, and the shared variant stages padded 32×32 input tiles
in threadgroup memory. They require aligned dimensions.

| Function | Median ms | MPS time / own time | MLX time / own time |
| --- | ---: | ---: | ---: |
| MPS | 45.8813 | 1.00× | 0.97× |
| MLX | 44.4741 | 1.03× | 1.00× |
| `simdgroup_8x8` | 141.0564 | 0.33× | 0.32× |
| `simdgroup_32x32` | 88.9133 | 0.52× | 0.50× |
| `simdgroup_32x32_shared` | 55.3775 | 0.83× | 0.80× |

## What else is worth experimenting with?

`simdgroup_matrix` and TensorOps are compute primitives. The rest of GEMM
performance depends heavily on feeding them and arranging work:

1. **Tile shape, SIMD-group count, and occupancy.** Tune M/N/K tile sizes
   together. A larger tile reuses more inputs but can increase register use or
   reduce active threadgroups. Apple specifically recommends tuning these
   dimensions in its [M5 TensorOps talk](https://developer.apple.com/videos/play/tech-talks/111432/).
2. **K-loop scheduling.** Insert barriers at a chosen K cadence so cooperating
   SIMD groups stay near the same input tiles. The synchronized functions here
   test 128, 256, 512, and 1024. This was the largest gain observed so far.
3. **Memory staging and fragment reuse.** Load A/B cooperatively into padded
   threadgroup memory, keep output fragments in registers, and reuse each loaded
   A/B fragment across multiple output fragments. `simdgroup_32x32_shared`
   demonstrates one modest version of this; further layouts are open.
4. **Threadgroup traversal.** Test Morton, swizzled raster, or another locality
   order. Morton did not help these variants here. MLX uses a smaller tile
   swizzle for the inferred NAX route, so that is a separate experiment.
5. **Split K.** Give independent threadgroups parts of a long reduction and
   combine partial outputs. This adds a temporary buffer and reduction pass,
   and is most promising when M/N provide too few output tiles. MLX has both
   NAX and older Steel split-K routes.
6. **Precision and data format.** FP16/BF16 and relaxed FP32 may run faster,
   but they change the numerical contract. Quantized GEMM adds dequantization
   and can use cooperative tensor operands. Compare each setting with an
   appropriate accuracy metric and label it separately from strict FP32.
7. **Fusion and shape dispatch.** Fuse bias or activation into the live
   cooperative accumulator for a real network workload. Choose different
   variants for large square GEMM, skinny GEMM, and irregular edges rather than
   expecting one tile to win everywhere. This benchmark currently tests only
   the plain 4096³ case by default.

Apple's [M5 talk](https://developer.apple.com/videos/play/tech-talks/111432/)
and [MPP Programming Guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)
explain tile sizing, K synchronization, traversal, cooperative tensors, and
GPU profiling. The next performance evidence should include an Xcode Metal
trace with GPU execution time, occupancy, and cache counters.

## How MLX implements this workload

The following describes the inspected **MLX v0.32.1 source**, and the precise
kernel selected on this machine is an inference from its dispatch code rather
than a captured GPU trace.

- [`matmul.cpp`](https://github.com/ml-explore/mlx/blob/v0.32.1/mlx/backend/metal/matmul.cpp)
  selects the NAX route when the GPU supports it and FP32 TF32 mode is enabled.
  [`MLX_ENABLE_TF32` defaults to 1](https://github.com/ml-explore/mlx/blob/v0.32.1/mlx/utils.h).
  The 4096³ shape does not meet its NAX split-K threshold, so it goes through
  the regular NAX path.
- For the detected M5 GPU architecture (`applegpu_g17g`), that regular route
  uses a 128×128×512 block, a 4×4 SIMD-group arrangement, and a two-row
  threadgroup swizzle. These are the v0.32.1 dispatch defaults for architecture
  suffix `g`, not measurements of hardware utilization.
- [`gemm_nax.h`](https://github.com/ml-explore/mlx/blob/v0.32.1/mlx/backend/metal/kernels/steel/gemm/gemm_nax.h)
  uses tiled loads and a K loop with synchronization.
  [`nax.h`](https://github.com/ml-explore/mlx/blob/v0.32.1/mlx/backend/metal/kernels/steel/gemm/nax.h)
  wraps Metal 4 `matmul2d` in 16×32×16 SIMD-group fragments. Its descriptor
  enables relaxed precision. The fastest local kernel here explicitly disables
  relaxed precision, so the timing comparison has a precision-policy difference
  that deserves separate numerical analysis before a broad claim.
- When NAX is unavailable or FP32 TF32 mode is disabled, dispatch can select
  older Steel GEMM. Its [`gemm.h`](https://github.com/ml-explore/mlx/blob/v0.32.1/mlx/backend/metal/kernels/steel/gemm/gemm.h)
  and [`mma.h`](https://github.com/ml-explore/mlx/blob/v0.32.1/mlx/backend/metal/kernels/steel/gemm/mma.h)
  stage tiles in threadgroup memory and compute with `simdgroup_matrix`.

An experimental `relaxed_precision = true` local TensorOps variant failed the
existing MPS agreement test at `7×13×5` (two mismatches, maximum absolute
error 0.005826), so it was not registered. This shows why the precision choice
needs explicit reporting. It does not establish which implementation is more
accurate against a high-precision reference.

## Reproduce and extend

```sh
./build/bin/gemm list
./build/bin/gemm bench --kernel tensorops_sync --iterations 5 \
  --output sync_sweep_1.csv
./build/bin/gemm bench --kernel simdgroup --iterations 5 \
  --output simdgroup_sweep_1.csv
./build/bin/gemm bench --function tensorops_sync_k256 --iterations 10 \
  --output sync_k256_10_samples.csv
./build/bin/gemm bench --smoke --iterations 3 \
  --output smoke_after_change.csv
```

Custom output names are saved in `outputs/`; they must be plain `.csv`
filenames. The default benchmark CSV name is reused on each run. Change
[`gemm/params.h`](gemm/params.h) to expand the measured shape suite. Re-run
the correctness smoke suite whenever a shader changes.
