#include <metal_stdlib>

using namespace metal;

struct GemmShape
{
  uint M;
  uint N;
  uint K;
};

template <uint BN, uint BM, uint BK, uint TM>
kernel void matmul_1D_coarsening(
    device const float* A [[buffer(0)]],
    device const float* B  [[buffer(1)]],
    device float* C        [[buffer(2)]],
    device const GemmShape& params [[buffer(3)]],
    uint2 block_pos [[ threadgroup_position_in_grid ]],
    uint2 thread_pos [[ thread_position_in_threadgroup ]]
) {
    const uint M = params.M;
    const uint N = params.N;
    const uint K = params.K;

    // allocate shared memory
    threadgroup float A_shmem[BM * BK];
    threadgroup float B_shmem[BK * BN];

    int tid = thread_pos.x;

    // some convenient variables
    int bCol = block_pos.x;
    int bRow = block_pos.y;

    // Row and column of output C
    int tCol = tid % BN;
    int tRow = tid / BN;

    // advance pointer
    A += bRow * BM * K;
    B += bCol * BN;
    C += bRow * BM * N + bCol * BM;

    // Row and column of tile A and B
    const int tileColA = tid % BK;
    const int tileRowA = tid / BK;
    const int tileColB = tid % BN;
    const int tileRowB = tid / BN;

    // each thread will compute TM elements
    float sum[TM] = {0.f};

    for (uint ph = 0; ph < K; ph += BK) {
        // populate the SMEM caches (same as before)
        if ((tileRowA + bRow * BM < M) && (ph + tileColA < K)) {
            A_shmem[tileRowA * BK + tileColA] = A[tileRowA * K + tileColA];
        } else {
            A_shmem[tileRowA * BK + tileColA] = 0.f;
        }
        if ((ph + tileRowB < K) && (tileColB + bCol * BN < N)) {
            B_shmem[tileRowB * BN + tileColB] = B[tileRowB * N + tileColB];
        } else {
            B_shmem[tileRowB * BN + tileColB] = 0.f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // calculate per thread result
        for (uint k = 0; k < BK; ++k) {
            float Btmp = B_shmem[k * BN + tCol];
            for (uint i = 0; i < TM; ++i) {
                sum[i] += A_shmem[(tRow * TM + i) * BK + k] * Btmp;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // advance pointer
        A += BK;
        B += BK * N;
    }

    // populate result
    for (uint i = 0; i < TM; ++i) {
        uint innerRow = tRow * TM + i;
        uint innerCol = tCol;
        if ((innerRow + bRow * BM < M) && (innerCol + bCol * BN < N)) {
            C[innerRow * N + innerCol] = sum[i];
        }
    }
}

template [[ host_name("matmul_1D_coarsening_mn64k8") ]]
kernel void matmul_1D_coarsening<64, 64, 8, 8>(
    device const float* A [[buffer(0)]],
    device const float* B  [[buffer(1)]],
    device float* C        [[buffer(2)]],
    device const GemmShape& params [[buffer(3)]],
    uint2 block_pos [[ threadgroup_position_in_grid ]],
    uint2 thread_pos [[ thread_position_in_threadgroup ]]
);
