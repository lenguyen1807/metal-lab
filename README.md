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
./build/bin/gemm mps --smoke
./build/bin/gemm naive --smoke
ctest --test-dir build --output-on-failure
```

The smoke suite includes tiny, square, and irregular shapes through
`(M,N,K) = (257,263,255)`. It checks results against a CPU reference that
accumulates in double precision. The check uses
`abs(actual - expected) <= 1e-3 + 1e-3 * abs(expected)`.

## Benchmark

```sh
./build/bin/gemm mps
./build/bin/gemm naive
```

The default suite includes large shapes; the naive kernel can take a long time.
Each run writes `outputs/<kernel>.csv`. Smoke runs write
`outputs/<kernel>_smoke.csv`. New generated CSVs are ignored by Git.

Both implementations compute row-major, non-transposed FP32
`C = A @ B` (`alpha=1`, `beta=0`) from identical preallocated
`MTLStorageModeShared` buffers. The MPS adapter in `gemm/mps_gemm.mm` bridges
the existing metal-cpp objects to Objective-C MPS objects. It uses packed
`columns * sizeof(float)` row strides so the input layout matches the custom
kernel. MPS can recommend a different row stride for best performance; that is
a separate experiment, not part of this comparison.

The reported time is `GPUEndTime - GPUStartTime` for one completed command
buffer after one warm-up. It includes all GPU work MPS encodes into that buffer.
It excludes input generation, buffer allocation, encoding, submission, and the
CPU wait. The harness reports the mean of 20 runs. Tiny smoke timings are useful
for correctness checks, not performance claims. The default suite checks CPU
correctness for shapes whose three dimensions are at most 1024; unvalidated
rows have `nan` in the error column.

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
