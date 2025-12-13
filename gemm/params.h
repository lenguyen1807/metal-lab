#pragma once

#include <cstdint>
#include <vector>

constexpr int PFIRST = 256;
constexpr int PLAST = 3840;
constexpr int PINC = 128;

// https://github.com/philipturner/metal-benchmarks?tab=readme-ov-file#operations-per-second
constexpr float M2_GPU_GHZ = 1.398f;

constexpr int BENCHMARK_TIME = 20;
constexpr float EQUAL_EPSILON = 1e-5f;

struct MatmulParams
{
  uint32_t M;
  uint32_t N;
  uint32_t K;
};

// A standard suite of benchmark shapes
const std::vector<MatmulParams> BENCHMARK_SHAPES = {
    // 1. Powers of 2 - Square (Baseline & Cache/Tiling Behavior)
    // throughput.
    {512, 512, 512},
    {1024, 1024, 1024},
    {2048, 2048, 2048},
    {4096, 4096, 4096},

    // 2. LLM FFN Layers - Compute-Bound (Large K)
    // (batch * seq_len, hidden_dim) @ (hidden_dim, ffn_dim)
    {2048, 11008, 4096},
    {2048, 4096, 11008},

    // 3. LLM Attention Layers - Mixed Workloads
    // QK^T: (batch * num_heads, seq_len, head_dim) @ (batch * num_heads,
    // head_dim, seq_len)
    // This is a batched GEMM. For a single head, it's (seq_len, head_dim) @
    // (head_dim, seq_len)
    {4096, 4096, 128},
    {2048, 2048, 128},

    // 4. Memory-Bound - "Skinny" Matrices (Small K)
    {4096, 4096, 16},
    {4096, 4096, 32},
    {4096, 4096, 64},

    // 5. Sizes that are NOT multiples of tile size (e.g., 32 or 64)
    {1000, 1000, 1000},
    {2050, 2050, 130},
    {4097, 127, 4097}};
