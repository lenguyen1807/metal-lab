# Metal GEMM lab

A small C++20 and Metal lab for learning GEMM on Apple GPUs. The current
experiment compares a handwritten FP32 kernel with Apple's
`MPSMatrixMultiplication` on the same row-major buffers. MLX is not required.

## Build

Use macOS, Xcode with the Metal toolchain, and CMake 3.20 or newer. The project
uses AppleClang and Apple's official [metal-cpp](https://github.com/apple/metal-cpp)
as a pinned Git submodule. The current pin is
`27c4382b7151d55a51692cdcb27aaa98752240de`, which includes `MTL4` headers.

```sh
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
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
`(M,N,K) = (257,263,255)`. It runs MPS and each custom kernel once on the
same inputs, then compares every output element. The check uses
`abs(custom - MPS) <= 1e-3 + 1e-3 * abs(MPS)` and rejects non-finite values.
This establishes agreement with MPS; it is not an independent proof that MPS
or the shared input setup is correct.

## Benchmark

```sh
./build/bin/gemm list
./build/bin/gemm bench
./build/bin/gemm bench --smoke
./build/bin/gemm bench --kernel naive --iterations 5
```

`bench` runs MPS first and then every registered custom kernel for each shape.
`--kernel NAME` selects custom variants; MPS remains the baseline. The default
suite includes large shapes, so the naive kernel can take a while. The default
is 10 timing samples per variant and shape; `--iterations` changes that count.
The smoke suite uses the six smaller test shapes. Results go to one file per
GPU, such as `outputs/bench_apple_m5.csv` or
`outputs/bench_smoke_apple_m5.csv`. New generated CSVs are ignored by Git.

Each shape gets one table with `median ms`, `GFLOP/s`, and `vs MPS`.
`GFLOP/s = 2MNK / (median_ms * 10^6)`. `vs MPS` is
`MPS_median_ms / kernel_median_ms`, equivalent to the throughput ratio for the
same shape. Thus `0.20x` means one fifth of MPS throughput. Rows use
`baseline`, `ok`, or `unsupported` as their status.

Both implementations compute packed row-major, non-transposed FP32
`C = A @ B` (`alpha=1`, `beta=0`) from identical preallocated
`MTLStorageModeShared` buffers. The MPS adapter in `gemm/mps_gemm.mm` bridges
the existing metal-cpp objects to Objective-C MPS objects. It uses packed
`columns * sizeof(float)` row strides so the input layout matches the custom
kernel. MPS can recommend a different row stride for best performance. General
leading dimensions and an NT experiment are later, separate comparisons: NT
changes the logical operation, while row stride describes physical storage.

The native kernel registry in `gemm/kernel.cpp` assigns each variant its shader
function, supported shapes, launch geometry, and encoding callback. The
benchmark does not construct a native kernel's grid or bind its arguments.

The reported time is `GPUEndTime - GPUStartTime` for one completed command
buffer after one warm-up. It includes all GPU work MPS encodes into that buffer.
It excludes input generation, buffer allocation, encoding, submission, and the
CPU wait. Each custom kernel is compared with the MPS output after its warm-up;
the comparison is outside the timing loop. The harness reports the median of
the requested samples. Tiny smoke timings are useful for wiring checks, not
performance claims.

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
