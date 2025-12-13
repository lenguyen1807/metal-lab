# A journey to ~2.84 TFLOPs on my M2 MacBook

This is a diary of my journey to write a fast single-floating point (FP32) matrix multiplication (SGEMM) kernel on my Apple M2 laptop. Since I don't have an NVIDIA card lying around, I'm using Apple's **Metal** API instead of CUDA. The core ideas are the same: making GEMM as fast as possible. 

The theoretical FP32 peak of a 8-core M2 is **~2.84 TFLOPs** (or 2840 GFLOPS) [^1] [^2]. Can we even get close? Let's find out.

This repo is for anyone who wants to learn GPU optimization but don't have NVIDA GPU like me. The code is (a little bit) clean, and built with modern C++ so we don't leak memory all over the place (and shoot our foot in place).

## How fast are we so far?

| Kernel | Best performance (GFLOPS) | % of peak performance (2840 GLOPS) |
|--------|---------------------------|---|
| `naive` | $\approx 178$ | $\approx 6.26$% |
| `tile_16` | $\approx 269$ | $\approx 9.12$% |
| `tile_32` | $\approx 195$ | $\approx 6.86$% |
| `tile_threads` | $\approx 359$ | $\approx 12.6$% |
| `tile_simdgroup` | $\approx 421$ | $\approx 14.8$% |

We will conduct benchmarking on vary matrix sizes (`M x N x K`) to represent real-world scenarios then the final GLFOPS is the mean of all benchmarking tests.
- **Powers of 2 - Square** (for baseline): `M=N=K` and vary from `512` to `4096`.
- **FFN Layers** (compute-bound): Simulates the feed-forward networks in Transformers (e.g., Llama, GPT). These are typically compute-bound due to the large inner dimension (K) (`K >> M` and `K >> N`).
- **Attention Layers**: small `K`. For a signel attention head, the GEMM is `(seq_len, head_dim)` and `head_dim` is small compared with `seq_len`.
- **Skinny Matrices**: with really small `K`. These stress memory bandwidth. The kernel spends more time loading data than computing. GFLOPS will be significantly lower here.
- **Non-ideal size**: test the kernel's handling of edge cases and non-ideal dimensions.

## Get it running

>[!IMPORTANT]
>You'll need **a Mac** with an **M-series chip**.

### 1. Installation

If you don't have [Homebrew](https://brew.sh/), get it. Then:
```bash
brew install cmake make llvm@20
```

>[!NOTE]
>We need `llvm`'s clang because Apple's default one can be a bit... quirky. Note that `llvm` newest version (21) cannot work on our code properly so we choose version 20.

### 2. Build the thing

Pop open a terminal and run these:
```bash
# Make a home for the benchmark results
mkdir -p outputs

# Let CMake do its magic. This points to the new clang we just installed.
cmake -S . -B build -G "Unix Makefiles" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=$(brew --prefix llvm@20)/bin/clang \
      -DCMAKE_CXX_COMPILER=$(brew --prefix llvm@20)/bin/clang++

# Fire the lasers!
cmake --build build
```
If that all worked, you'll have a shiny new executable at `build/bin/gemm`.

## Run the benchmark

```bash
# It's easy, just tell it which kernel to run
./build/bin/gemm naive
```

The program will spit out performance numbers to your console and also save a detailed `.csv` file in the `outputs` folder. This is the good stuff you can use to make pretty graphs in Python.

**Available kernels:**
- `naive`: The humble beginning. 
- `tile_16` and `tile_32`: Tiling kernel with different tile sizes.
- `tile_threads`: Tiling kernel with more work on threads.
- `tile_simdgroup`: Tiling kernel with `simdgroup` (metal intrinsics).

## The Optimization Checklist

This project is structured so you can follow the optimization journey step-by-step. Each new kernel will be a separate file, building on the lessons of the last.

- [x] **Chapter 0: The Setup** Build a solid, memory-safe C++ framework with a real benchmark harness. No segfaults allowed.
- [x] **Chapter 1: The Tiling** The first real optimization. Use that sweet, sweet shared memory or SMEM (`threadgroup` in Metal, `__shared__` in CUDA) to stop hitting DRAM so much. This is where we should see the first big performance jump.
- [x] **Chapter 2: More Work, Less Laziness (Register Tiling)** Make each thread compute a small 2x2 or 4x4 block of the output matrix. This increases register reuse and hides instruction latency.
- [x] **Chapter 3: Embracing the Hardware (SIMD-group Matrix Primitives)** This is the game-changer. We stop thinking in scalar operations (a * b) and start thinking in matrices. We'll use Metal's `simdgroup` to command the M2's matrix acceleration hardware (Apple's equivalent of Tensor Cores).

## Analysis

### Chapter 0: Naive GEMM

### Chapter 1: The tiling

#### Why we need tiling ?

> [!CAUTION]
> TODO

#### Why is `tile_16` Faster Than `tile_32`?

This result is counter-intuitive at first. A larger tile size like 32x32 should mean more data reuse within the fast `threadgroup` memory (or *shared memory*), which is usually good for performance. However, it's slower. Why?

The answer is **Occupancy**.

1.  **What is Occupancy?** Occupancy is the ratio of active threadgroups (or warps) to the maximum number of threadgroups that can run on a single GPU compute unit (CU) (or an SM in CUDA device). High occupancy is critical for hiding memory latency. When one group of threads is stalled waiting for data to arrive from the slow device memory (DRAM), the GPU scheduler can switch to another *resident* group and keep the compute units busy.

2.  **Resource Limits:** A CU has a fixed amount of resources, including registers and, most importantly for this case, `threadgroup` memory.
- `tile_16` kernel:
      - Threadgroup size: 16x16 = 256 threads.
      - `threadgroup` memory used: `(16*16 + 16*16) * 4 bytes = 2048 bytes`.
- `tile_32` kernel:
      - Threadgroup size: 32x32 = 1024 threads.
      - `threadgroup` memory used: `(32*32 + 32*32) * 4 bytes = 8192 bytes`.

3.  **The Bottleneck:** The M2 GPU's CUs have a limited amount of `threadgroup` memory (32 KB) [^3]. The `tile_32` kernel's 8KB memory footprint is significant. If a single threadgroup consumes too large a chunk of the CU's available memory, the scheduler cannot fit as many *concurrent* threadgroups onto that CU.

With `tile_32`, you might only be able to fit one or two threadgroups per CU, leading to low occupancy. If those few groups stall on a memory read, there are no other resident groups to switch to, and the expensive ALU units sit idle.

The `tile_16` kernel, with its much smaller 2KB footprint, allows many more threadgroups to be resident on the CU simultaneously. This gives the scheduler a large pool of work to choose from, effectively hiding memory latency and keeping the hardware busy.horrors.

### Chapter 2: More work on threads

> [!CAUTION]
> TODO

### Chapter 3: SIMD Tilegroup

> [!CAUTION]
> TODO

## Resources

- [siboehm's CUDA Matrix Optimization](https://siboehm.com/articles/22/CUDA-MMM)
- [OpenCL SGEMM Tutorial](https://cnugteren.github.io/tutorial/pages/page1.html)
- [Cuda C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [metal_performance_testing by bkvogel](https://github.com/bkvogel/metal_performance_testing)
- [metal_flash_attention by philipturner](https://github.com/philipturner/metal-flash-attention/tree/main)

[^1]: https://www.cpu-monkey.com/en/cpu-apple_m2_8_gpu
[^2]: https://github.com/philipturner/metal-benchmarks
[^3]: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf