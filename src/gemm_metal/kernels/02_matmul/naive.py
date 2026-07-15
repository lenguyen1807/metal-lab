from __future__ import annotations

import mlx.core as mx

from gemm_metal.kernels.utils import check_matmul_inputs

BLOCK_SIZE = 16

matmul_naive_kernel = mx.fast.metal_kernel(
    name="matmul_naive",
    input_names=["A", "B"],
    output_names=["C"],
    source=r"""
        // CUDA mental model:
        //   blockIdx.{x,y}  ~= threadgroup_position_in_grid.{x,y}
        //   threadIdx.x     ~= thread_position_in_threadgroup.x
        // MLX's Metal launch uses grid=(total threads), not grid=(threadgroups).

        uint bx = threadgroup_position_in_grid.x;
        uint by = threadgroup_position_in_grid.y;

        uint tid = thread_position_in_threadgroup.x;
        uint tx = tid % BLOCK_SIZE;
        uint ty = tid / BLOCK_SIZE;

        uint col = bx * BLOCK_SIZE + tx;
        uint row = by * BLOCK_SIZE + ty;

        if (row < M && col < N) {
            float sum = 0.0f;
            for (uint p = 0; p < K; ++p) {
                sum += float(A[row * K + p]) * float(B[p * N + col]);
            }
            C[row * N + col] = sum;
        }
    """,
)


def matmul_naive(A: mx.array, B: mx.array, M: int, N: int, K: int) -> mx.array:
    check_matmul_inputs(A, B, M, N, K)

    blocks_x = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    blocks_y = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    threads_per_group = BLOCK_SIZE * BLOCK_SIZE

    # Unlike CUDA's dim3 grid(blocks_x, blocks_y), MLX Metal expects the full
    # thread grid. With a 1D threadgroup of 256 threads, x must be
    # blocks_x * threads_per_group so Metal creates blocks_x threadgroups in x.
    grid_dim = (blocks_x * threads_per_group, blocks_y, 1)

    outputs = matmul_naive_kernel(
        inputs=[A, B],
        grid=grid_dim,
        threadgroup=(threads_per_group, 1, 1),
        template=[("M", M), ("N", N), ("K", K), ("BLOCK_SIZE", BLOCK_SIZE)],
        output_shapes=[(M, N)],
        output_dtypes=[A.dtype],
    )
    return outputs[0]
