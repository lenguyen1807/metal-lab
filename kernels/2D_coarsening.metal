#include <metal_stdlib>

using namespace metal;

struct GemmShape
{
  uint M;
  uint N;
  uint K;
};

template <uint BN, uint BM, uint BK, uint TM, uint TN>
kernel void matmul_2D_coarsening(
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

    // A thread is responsible for calculating TM*TN elements in the blocktile
    // Here is how we calculate total threads for each block tile
    const int nThreadsTile = (BM * BN) / (TM * TN);

    // some convenient variables
    int bCol = block_pos.x;
    int bRow = block_pos.y;

    // advance pointer
    A += bRow * BM * K;
    B += bCol * BN;
    C += bRow * BM * N + bCol * BN;

    // Row and column of the TM x TN output patch computed by this thread.
    int tCol = tid % (BN / TN);
    int tRow = tid / (BN / TN);

    // Row and column of tile A and B
    const int tileColA = tid % BK;
    const int tileRowA = tid / BK;
    const int tileColB = tid % BN;
    const int tileRowB = tid / BN;

    // for both As and Bs we want each load to span the full column-width, for
    // better GMEM coalescing (as opposed to spanning full row-width and iterating
    // across columns)
    const int strideA = nThreadsTile / BK;
    const int strideB = nThreadsTile / BN;

    // each thread will compute TM x TN elements
    float sum[TM * TN] = {0.f};
    // register caches for As and Bs
    float regA[TM] = {0.0};
    float regB[TN] = {0.0};

    // outer loop
    for (int ph = 0; ph < K; ph += BK) {
        // populate the SMEM caches (same as before)
        for (int offset = 0; offset < BM; offset += strideA) {
            int rowA = tileRowA + offset;
            int colA = tileColA;
            if ((rowA + bRow * BM < M) && (ph + colA < K)) {
                A_shmem[rowA * BK + colA] = A[rowA * K + colA];
            } else {
                A_shmem[rowA * BK + colA] = 0.f;
            }
        }
        for (int offset = 0; offset < BK; offset += strideB) {
            int rowB = tileRowB + offset;
            int colB = tileColB;
            if ((rowB + ph < K) && (colB + bCol * BN < N)) {
                B_shmem[rowB * BN + colB] = B[rowB * N + colB];
            } else {
                B_shmem[rowB * BN + colB] = 0.f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // calculate result
        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; ++i) {
                // load row of A tile to register
                regA[i] = A_shmem[(tRow * TM + i) * BK + k];
            }
            for (int i = 0; i < TN; ++i) {
                // load column of B tile to register
                regB[i] = B_shmem[k * BN + (tCol * TN + i)];
            }
            // now calculate result from register
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    sum[i * TN + j] += regA[i] * regB[j];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // advance pointer
        A += BK;
        B += BK * N;
    }

    // populate result
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            int innerRow = tRow * TM + i;
            int innerCol = tCol * TN + j;
            if ((innerRow + bRow * BM < M) &&
                (innerCol + bCol * BN < N)) {
                C[innerRow * N + innerCol] = sum[i * TN + j];
            }
        }
    }
}

template [[ host_name("matmul_2D_coarsening_mn128k8") ]]
kernel void matmul_2D_coarsening<128, 128, 8, 8, 8>(
    device const float* A [[buffer(0)]],
    device const float* B  [[buffer(1)]],
    device float* C        [[buffer(2)]],
    device const GemmShape& params [[buffer(3)]],
    uint2 block_pos [[ threadgroup_position_in_grid ]],
    uint2 thread_pos [[ thread_position_in_threadgroup ]]
);
