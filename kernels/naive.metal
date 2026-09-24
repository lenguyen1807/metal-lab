#include <metal_stdlib>

struct GemmShape
{
  uint M;
  uint N;
  uint K;
};

kernel void matmul_naive(device const float * A [[buffer(0)]],
                         device const float * B [[buffer(1)]],
                         device float * C       [[buffer(2)]],
                         device const GemmShape& params [[buffer(3)]],
                         uint2 block_pos [[ threadgroup_position_in_grid ]],
                         uint2 thread_pos [[ thread_position_in_threadgroup ]],
                         uint2 threads_per_group [[ threads_per_threadgroup ]])
{
    // Calculate row and col
    const uint j = block_pos.x * threads_per_group.x + thread_pos.x; // row
    const uint i = block_pos.y * threads_per_group.y + thread_pos.y; // col

    const uint M = params.M;
    const uint N = params.N;
    const uint K = params.K;

    if (i < M && j < N)
    {
        float sum = 0.f;
        for (uint p = 0; p < K; ++p)
        {
            sum += A[i * K + p] * B[p * N + j];
        }
        C[i * N + j] = sum;
    }
}
