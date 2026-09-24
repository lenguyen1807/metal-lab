#include <metal_stdlib>

struct MatmulParams
{
  uint M;
  uint N;
  uint K;
  float alpha;
  float beta;
  uint BLOCK_SIZE_X;
  uint BLOCK_SIZE_Y;
};

kernel void matmul_naive(device const float * A [[buffer(0)]],
                         device const float * B [[buffer(1)]],
                         device float * C       [[buffer(2)]],
                         device const MatmulParams& params [[buffer(3)]],
                         uint2 block_pos [[ threadgroup_position_in_grid ]],
                         uint2 thread_pos [[ thread_position_in_threadgroup ]])
{
    // Block index
    const uint block_x = block_pos.x; // CUDA: blockIdx.x
    const uint block_y = block_pos.y; // CUDA: blockIdx.y

    // Thread index
    const uint thread_x = thread_pos.x; // CUDA: threadIdx.x
    const uint thread_y = thread_pos.y; // CUDA: threadIdx.y

    // Calculate row and col
    // We use params.BLOCK_SIZE_X as block dimension
    const uint j = block_x * params.BLOCK_SIZE_X + thread_x; // row
    const uint i = block_y * params.BLOCK_SIZE_Y + thread_y; // col

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
        C[i * N + j] = params.alpha * sum + params.beta * C[i * N + j];
    }
}
