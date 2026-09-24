# Metal GEMM lab

A small FP32 GEMM workbench for Apple GPUs. It compares handwritten Metal
kernels with MPS and MLX GPU `matmul`. The current operation is packed,
row-major `C = A @ B` with neither input transposed (NN).

## Setup Metal

```bash
xcode-select --install
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
xcodebuild -downloadComponent MetalToolchain
```

## Build

Requires macOS, Xcode's Metal toolchain, CMake 3.20+, and
[MLX from Homebrew](https://formulae.brew.sh/formula/mlx). Apple
[metal-cpp](https://github.com/apple/metal-cpp) is pinned as a submodule at
`27c4382b7151d55a51692cdcb27aaa98752240de`; this pin includes Metal 4
headers.

```sh
brew install mlx
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DMLX_DIR="$(brew --prefix mlx)/share/cmake/MLX"
cmake --build build -j
```

## Run

```sh
./build/bin/gemm list
./build/bin/gemm bench --smoke --iterations 3
./build/bin/gemm bench
./build/bin/gemm bench --kernel naive --iterations 5
```

`bench` runs every registered custom kernel alongside both baselines.
`--kernel NAME` selects custom kernels; it never removes MPS or MLX.
`--smoke` uses six small and irregular shapes. The default suite contains
square shapes from 512³ through 4096³ and three 4096×4096 cases with small K;
edit [`gemm/params.h`](gemm/params.h) to change it. Timing uses 10 samples
per shape and implementation unless `--iterations` is given.

Every shape produces MPS, MLX, and custom rows with median milliseconds,
GFLOP/s, `vs MPS`, and `vs MLX`. GFLOP/s is `2MNK / (median_ms × 10⁶)`.
Each ratio is baseline time divided by row time, so `0.20x` means one fifth
of that baseline's throughput. The same rows go to
`outputs/bench_<gpu>.csv` (or `bench_smoke_<gpu>.csv`); generated CSVs are
ignored by Git.

## Correctness and timing

Before timing each shape, the harness runs MPS, MLX, and each supported custom
kernel once. It compares every MLX and custom output element with MPS using
`abs(result - MPS) <= 1e-3 + 1e-3 * abs(MPS)`; non-finite values fail.
Validation is outside the timing loop and error values are omitted from the
results. MPS agreement is the chosen reference, not an independent proof of
mathematical correctness. The smoke benchmark is the quick correctness check;
there is no separate test command.

All inputs contain identical FP32 values. MPS and custom kernels share packed
Metal buffers; MLX receives its own GPU arrays before timing. Reported times
are host `steady_clock` medians for **completed calls**: Metal calls include
command-buffer encoding, submission, and waiting; MLX calls include graph
creation, evaluation, and GPU synchronization. MLX allocates a result each
call, while Metal reuses output buffers. These numbers compare the current
calling paths, not isolated GPU kernel execution.

## Add a kernel

Add `kernels/<name>.metal`, then register its function, supported shapes,
dispatch plan, and encoder in [`gemm/kernel.cpp`](gemm/kernel.cpp).
The harness lets each kernel choose its own grid and bindings. Run
`./build/bin/gemm bench --kernel <name> --smoke` before the full suite.

The current custom path supports packed NN matrices. General row strides and
transposed layouts can be added as separate experiments. Metal objects on the
C++ side use `NS::SharedPtr`; the MPS bridge uses Objective-C++ ARC.
