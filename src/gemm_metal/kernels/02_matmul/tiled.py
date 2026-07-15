from __future__ import annotations

import mlx.core as mx

from gemm_metal.kernels.utils import check_matmul_inputs

matmul_tiled_kernel = mx.fast.metal_kernel(
    name="matmul_tiled",
    input_names=["A", "B"],
    output_names=["C"],
    source=r"""
        uint bx = threadgroup_position_in_grid.x;
        uint by = threadgroup_position_in_grid.y;

        uint tid = thread_position_in_threadgroup.x;
        uint tx = tid % TILE_WIDTH;
        uint ty = tid / TILE_WIDTH;

        uint col = bx * TILE_WIDTH + tx;
        uint row = by * TILE_WIDTH + ty;

        threadgroup float A_shared[TILE_WIDTH * TILE_WIDTH];
        threadgroup float B_shared[TILE_WIDTH * TILE_WIDTH];

        float sum = 0.0f;
        uint num_tiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

        for (uint tile = 0; tile < num_tiles; ++tile) {
            uint A_col = tile * TILE_WIDTH + tx;
            uint B_row = tile * TILE_WIDTH + ty;

            if (row < M && A_col < K) {
                A_shared[ty * TILE_WIDTH + tx] = float(A[row * K + A_col]);
            } else {
                A_shared[ty * TILE_WIDTH + tx] = 0.0f;
            }

            if (B_row < K && col < N) {
                B_shared[ty * TILE_WIDTH + tx] = float(B[B_row * N + col]);
            } else {
                B_shared[ty * TILE_WIDTH + tx] = 0.0f;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint k = 0; k < TILE_WIDTH; ++k) {
                sum += A_shared[ty * TILE_WIDTH + k] * B_shared[k * TILE_WIDTH + tx];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (row < M && col < N) {
            C[row * N + col] = sum;
        }
    """,
)


def matmul_tiled(
    A: mx.array,
    B: mx.array,
    M: int,
    N: int,
    K: int,
    tile_width: int,
) -> mx.array:
    check_matmul_inputs(A, B, M, N, K)
    if tile_width <= 0:
        raise ValueError(f"tile_width must be positive, got {tile_width}")

    blocks_x = (N + tile_width - 1) // tile_width
    blocks_y = (M + tile_width - 1) // tile_width
    threads_per_group = tile_width * tile_width
    grid_dim = (blocks_x * threads_per_group, blocks_y, 1)

    outputs = matmul_tiled_kernel(
        inputs=[A, B],
        grid=grid_dim,
        threadgroup=(threads_per_group, 1, 1),
        template=[("M", M), ("N", N), ("K", K), ("TILE_WIDTH", tile_width)],
        output_shapes=[(M, N)],
        output_dtypes=[A.dtype],
    )

    return outputs[0]


def matmul_tiled_16(A: mx.array, B: mx.array, M: int, N: int, K: int) -> mx.array:
    return matmul_tiled(A, B, M, N, K, 16)


def matmul_tiled_32(A: mx.array, B: mx.array, M: int, N: int, K: int) -> mx.array:
    return matmul_tiled(A, B, M, N, K, 32)
