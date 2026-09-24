# Metal GEMM lab

A small C++20 and Metal lab for learning GEMM on Apple GPUs. The current
experiment compares a handwritten FP32 kernel with Apple's
`MPSMatrixMultiplication` and MLX GPU `matmul` on the same row-major inputs.

## Build

Use macOS, Xcode with the Metal toolchain, CMake 3.20 or newer, and MLX from
Homebrew. The project uses AppleClang and Apple's official
[metal-cpp](https://github.com/apple/metal-cpp) as a pinned Git submodule.
The current pin is
`27c4382b7151d55a51692cdcb27aaa98752240de`, which includes `MTL4` headers.

```sh
brew install mlx
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DMLX_DIR="$(brew --prefix mlx)/share/cmake/MLX"
cmake --build build -j
```

`metal/definition.cpp` is the single translation unit that defines metal-cpp's
implementation symbols. The C++ host path currently uses Metal's original
command queue API; the newer headers make the Metal 4 API available for later
experiments.

## Check correctness

```sh
./build/bin/gemm test
ctest --test-dir build --output-on-failure
```

The test suite includes tiny, square, and irregular shapes through
`(M,N,K) = (257,263,255)`. It runs MPS, MLX GPU `matmul`, and each custom
kernel once on identical inputs, then compares every output element to MPS.
The check uses `abs(result - MPS) <= 1e-3 + 1e-3 * abs(MPS)` and rejects
non-finite values. This establishes agreement with MPS; it is not an
independent proof that MPS or the shared input setup is correct.

## Benchmark

```sh
./build/bin/gemm list
./build/bin/gemm bench
./build/bin/gemm bench --smoke
./build/bin/gemm bench --kernel naive --iterations 5
```

`bench` runs MPS, MLX, and every registered custom kernel for each shape.
`--kernel NAME` selects custom variants; both baselines remain. The default
suite includes large shapes, so the naive kernel can take a while. The default
is 10 timing samples per variant and shape; `--iterations` changes that count.
The smoke suite uses the six smaller test shapes. Results go to one file per
GPU, such as `outputs/bench_apple_m5.csv` or
`outputs/bench_smoke_apple_m5.csv`. New generated CSVs are ignored by Git.

Each shape gets one table with `median ms`, `GFLOP/s`, `vs MPS`, and `vs MLX`.
`GFLOP/s = 2MNK / (median_ms * 10^6)`. `vs MPS` is
`MPS_median_ms / kernel_median_ms`, equivalent to the throughput ratio for the
same shape; `vs MLX` uses the MLX time in the numerator. Thus `0.20x` means
one fifth of that baseline's throughput. Both MPS and MLX rows have
`baseline` status; custom rows have `ok` or `unsupported`.

All three implementations compute packed row-major, non-transposed FP32
`C = A @ B` (`alpha=1`, `beta=0`) from identical input values. MPS and the
native kernels share preallocated `MTLStorageModeShared` buffers. The MLX
adapter in `gemm/mlx_gemm.cpp` copies those values into MLX GPU arrays before
warmup and timing. The MPS adapter in `gemm/mps_gemm.mm` bridges metal-cpp
objects to Objective-C MPS objects. It uses packed
`columns * sizeof(float)` row strides so the input layout matches the custom
kernel. MPS can recommend a different row stride for best performance. General
leading dimensions and an NT experiment are later, separate comparisons: NT
changes the logical operation, while row stride describes physical storage.

The native kernel registry in `gemm/kernel.cpp` assigns each variant its shader
function, supported shapes, launch geometry, and encoding callback. The
benchmark does not construct a native kernel's grid or bind its arguments.

The reported time is host `steady_clock` time for one completed call after
warmup. MPS and native calls include command-buffer creation, encoding,
submission, and the wait. MLX calls include `matmul` graph creation,
evaluation, and GPU synchronization. Input generation and transfer are outside
the timing loop. MLX allocates a new result per call, while MPS and native
kernels reuse output buffers, so these are completed-call comparisons rather
than isolated GPU-kernel times. Every MLX and custom output is compared with
MPS after warmup and before timing; the comparison is outside the timing loop.
The harness reports the median of the requested samples. Tiny smoke timings
are useful for wiring checks, not performance claims.

## Object ownership

Retained metal-cpp objects use `NS::SharedPtr<T>` and `NS::TransferPtr(...)`:
the latter takes ownership of the +1 reference returned by `new`, `alloc/init`,
or `CreateSystemDefaultDevice`. `NS::RetainPtr(...)` is used when retaining an
autoreleased object, such as a command buffer, beyond a borrowed raw pointer.
Autorelease pools bound temporary Objective-C object lifetimes. The Objective-C++
MPS adapter uses ARC for its MPS objects. Raw pointers passed to encoders and
MPS are borrowed for the duration of the call; the owning C++ objects outlive
the completed command buffer.

This ownership pattern follows [Apple's metal-cpp guidance](https://developer.apple.com/metal/cpp/).
