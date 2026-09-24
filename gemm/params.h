#pragma once

#include <cstdint>
#include <vector>

constexpr int PFIRST = 256;
constexpr int PLAST = 3840;
constexpr int PINC = 128;

constexpr int BENCHMARK_TIME = 20;

struct MatmulParams
{
  uint32_t M;
  uint32_t N;
  uint32_t K;
  float alpha;
  float beta;
  uint32_t BLOCK_SIZE_X;
  uint32_t BLOCK_SIZE_Y;
};

// Structure to define a GEMM problem shape
struct GemmShape
{
  uint32_t M, N, K;
};

// A standard suite of benchmark shapes
const std::vector<GemmShape> BENCHMARK_SHAPES = {
    // 1. Powers of 2 - Square (Baseline & Cache/Tiling Behavior)
    // throughput.
    // Performance should be high and stable here.
    {512, 512, 512},
    {1024, 1024, 1024},
    {2048, 2048, 2048},
    {4096, 4096, 4096},

    // 2. LLM FFN Layers - Compute-Bound (Large K)
    // Simulates the feed-forward networks in Transformers (e.g., Llama, GPT).
    // These are typically compute-bound due to the large inner dimension (K).
    // A (batch * seq_len, hidden_dim) @ (hidden_dim, ffn_dim) problem.
    {2048, 11008, 4096},  // Llama-7B FFN up-projection
    {2048, 4096, 11008},  // Llama-7B FFN down-projection

    // 3. LLM Attention Layers - Mixed Workloads
    // QK^T: (batch * num_heads, seq_len, head_dim) @ (batch * num_heads,
    // head_dim, seq_len)
    // This is a batched GEMM. For a single head, it's (seq_len, head_dim) @
    // (head_dim, seq_len)
    // Often memory-bound for small head_dim.
    {4096, 4096, 128},  // QK^T for long sequence length, common head_dim=128
    {2048, 2048, 128},  // QK^T for medium sequence length

    // 4. Memory-Bound - "Skinny" Matrices (Small K)
    // These stress memory bandwidth. The kernel spends more time loading data
    // than computing. Your GFLOPS will be significantly lower here.
    {4096, 4096, 16},
    {4096, 4096, 32},
    {4096, 4096, 64},

    // 5. Sizes that are NOT multiples of tile size (e.g., 32 or 64)
    // These test the kernel's handling of edge cases and non-ideal dimensions.
    // Performance degradation should be graceful, not catastrophic.
    {1000, 1000, 1000},
    {2050, 2050, 130},
    {4097, 127, 4097}};
