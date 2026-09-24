#include <metal_stdlib>

using namespace metal;

struct GemmShape
{
  uint M;
  uint N;
  uint K;
};

template <uint BLOCK_N, uint BLOCK_K, uint BLOCK_M>
kernel void matmul_tiling(device const float* A [[buffer(0)]],
                        device const float* B  [[buffer(1)]],
                        device float* C        [[buffer(2)]],
                        device const GemmShape& params [[buffer(3)]],
                        uint2 block_pos [[ threadgroup_position_in_grid ]],
                        uint2 thread_pos [[ thread_position_in_threadgroup ]])
{
    // allocate shared memory
    threadgroup float A_shmem[BLOCK_M * BLOCK_K];
    threadgroup float B_shmem[BLOCK_K * BLOCK_N];

    // some convenient variables
    int bCol = block_pos.x;
    int bRow = block_pos.y;
    int tCol = thread_pos.x;
    int tRow = thread_pos.y;

    const uint M = params.M;
    const uint N = params.N;
    const uint K = params.K;

    /*
    * we advance pointer to current row (for A)
    * or current column (for B)
    * and then compute tile from there
    */
    A += bRow * BLOCK_M * K;
    B += bCol * BLOCK_N;
    C += bRow * BLOCK_M * N + bCol * BLOCK_N;

    float sum = 0.f;

    // the outer loop advances A along the columns and B along
    for (uint ph = 0; ph < K; ph += BLOCK_K) {
        // A tile is BLOCK_M x BLOCK_K, so only the first BLOCK_K columns of
        // the BLOCK_M x BLOCK_N threadgroup store it. B is BLOCK_K x BLOCK_N.
        if (tCol < BLOCK_K) {
            if ((tRow + bRow * BLOCK_M < M) && (ph + tCol < K)) {
                A_shmem[tRow * BLOCK_K + tCol] = A[tRow * K + tCol];
            } else {
                A_shmem[tRow * BLOCK_K + tCol] = 0.f;
            }
        }

        if (tRow < BLOCK_K) {
            if ((ph + tRow < K) && (tCol + bCol * BLOCK_N < N)) {
                B_shmem[tRow * BLOCK_N + tCol] = B[tRow * N + tCol];
            } else {
                B_shmem[tRow * BLOCK_N + tCol] = 0.f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // compute dot product of tile
        for (uint k = 0; k < BLOCK_K; ++k) {
            sum += A_shmem[tRow * BLOCK_K + k] * B_shmem[k * BLOCK_N + tCol];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // advance pointers onto next chunk
        A += BLOCK_K;
        B += BLOCK_K * N;
    }

    if ((tRow + bRow * BLOCK_M < M) && (tCol + bCol * BLOCK_N < N)) {
        C[tRow * N + tCol] = sum;
    }
}

template [[ host_name("matmul_tiling_mnk16") ]]
kernel void matmul_tiling<16, 16, 16>(device const float* A [[buffer(0)]],
                        device const float* B  [[buffer(1)]],
                        device float* C        [[buffer(2)]],
                        device const GemmShape& params [[buffer(3)]],
                        uint2 block_pos [[ threadgroup_position_in_grid ]],
                        uint2 thread_pos [[ thread_position_in_threadgroup ]]);

template [[ host_name("matmul_tiling_mnk32") ]]
kernel void matmul_tiling<32, 32, 32>(device const float* A [[buffer(0)]],
                        device const float* B  [[buffer(1)]],
                        device float* C        [[buffer(2)]],
                        device const GemmShape& params [[buffer(3)]],
                        uint2 block_pos [[ threadgroup_position_in_grid ]],
                        uint2 thread_pos [[ thread_position_in_threadgroup ]]);

template [[ host_name("matmul_tiling_mn32k16") ]]
kernel void matmul_tiling<32, 16, 32>(device const float* A [[buffer(0)]],
                        device const float* B  [[buffer(1)]],
                        device float* C        [[buffer(2)]],
                        device const GemmShape& params [[buffer(3)]],
                        uint2 block_pos [[ threadgroup_position_in_grid ]],
                        uint2 thread_pos [[ thread_position_in_threadgroup ]]);
