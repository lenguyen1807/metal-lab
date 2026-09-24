#pragma once

#include <cstdint>
#include <vector>

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
