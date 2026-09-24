#pragma once

#include <cstdint>
#include <vector>

struct MatmulParams {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    float alpha;
    float beta;
    uint32_t BLOCK_SIZE_X;
    uint32_t BLOCK_SIZE_Y;
};

// Structure to define a GEMM problem shape
struct GemmShape {
    uint32_t M, N, K;
};

// A standard suite of benchmark shapes
// const std::vector<GemmShape> BENCHMARK_SHAPES = {{512, 512, 512},
//                                                  {1024, 1024, 1024},
//                                                  {2048, 2048, 2048},
//                                                  {4096, 4096, 4096},
//                                                  {8192, 8192, 8192}};

// we will focus on optimizing for one shape only
const std::vector<GemmShape> BENCHMARK_SHAPES = {{4096, 4096, 4096}};
