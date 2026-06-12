"""GEMM kernels."""

from __future__ import annotations

import mlx.core as mx


def mlx_matmul(a: mx.array, b: mx.array) -> mx.array:
    return a @ b


matmul_naive_kernel_block_style = mx.fast.metal_kernel(
    name="matmul_naive_block_style",
    input_names=["A", "B"],
    output_names=["C"],
    source=r"""
        uint block_x = threadgroup_position_in_grid.x;
        uint block_y = threadgroup_position_in_grid.y;

        uint thread_x = thread_position_in_threadgroup.x;
        uint thread_y = thread_position_in_threadgroup.y;

        uint col = block_x * BLOCK_SIZE_X + thread_x;
        uint row = block_y * BLOCK_SIZE_Y + thread_y;

        if (row < M && col < N) {
            float sum = 0.0f;

            for (uint p = 0; p < K; ++p) {
                sum += float(A[row * K + p]) * float(B[p * N + col]);
            }

            C[row * N + col] = T(sum);
        }
    """,
)


def matmul_naive_block_style(
    A: mx.array,
    B: mx.array,
    block_x: int = 16,
    block_y: int = 16,
) -> mx.array:
    M, K = A.shape
    _, N = B.shape

    grid = (
        ((N + block_x - 1) // block_x) * block_x,
        ((M + block_y - 1) // block_y) * block_y,
        1,
    )

    (C,) = matmul_naive_kernel_block_style(
        inputs=[A, B],
        output_shapes=[(M, N)],
        output_dtypes=[A.dtype],
        template=[
            ("T", A.dtype),
            ("M", M),
            ("N", N),
            ("K", K),
            ("BLOCK_SIZE_X", block_x),
            ("BLOCK_SIZE_Y", block_y),
        ],
        grid=grid,
        threadgroup=(block_x, block_y, 1),
    )
    return C


matmul_tiled_kernel_block_style = mx.fast.metal_kernel(
    name="matmul_tiled_block_style",
    input_names=["A", "B"],
    output_names=["C"],
    source=r"""
        uint block_x = threadgroup_position_in_grid.x;
        uint block_y = threadgroup_position_in_grid.y;

        uint thread_x = thread_position_in_threadgroup.x;
        uint thread_y = thread_position_in_threadgroup.y;

        uint col = block_x * BLOCK_SIZE + thread_x;
        uint row = block_y * BLOCK_SIZE + thread_y;

        threadgroup float A_tile[BLOCK_SIZE][BLOCK_SIZE];
        threadgroup float B_tile[BLOCK_SIZE][BLOCK_SIZE];

        float sum = 0.0f;

        for (uint tile = 0; tile < K; tile += BLOCK_SIZE) {
            uint a_col = tile + thread_x;
            uint b_row = tile + thread_y;

            A_tile[thread_y][thread_x] =
                (row < M && a_col < K) ? float(A[row * K + a_col]) : 0.0f;
            B_tile[thread_y][thread_x] =
                (b_row < K && col < N) ? float(B[b_row * N + col]) : 0.0f;

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint p = 0; p < BLOCK_SIZE; ++p) {
                sum += A_tile[thread_y][p] * B_tile[p][thread_x];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (row < M && col < N) {
            C[row * N + col] = T(sum);
        }
    """,
)


def matmul_tiled_block_style(A: mx.array, B: mx.array, block_size: int) -> mx.array:
    M, K = A.shape
    _, N = B.shape

    grid = (
        ((N + block_size - 1) // block_size) * block_size,
        ((M + block_size - 1) // block_size) * block_size,
        1,
    )

    (C,) = matmul_tiled_kernel_block_style(
        inputs=[A, B],
        output_shapes=[(M, N)],
        output_dtypes=[A.dtype],
        template=[
            ("T", A.dtype),
            ("M", M),
            ("N", N),
            ("K", K),
            ("BLOCK_SIZE", block_size),
        ],
        grid=grid,
        threadgroup=(block_size, block_size, 1),
    )
    return C


def matmul_tiled_16(A: mx.array, B: mx.array) -> mx.array:
    return matmul_tiled_block_style(A, B, block_size=16)


def matmul_tiled_32(A: mx.array, B: mx.array) -> mx.array:
    return matmul_tiled_block_style(A, B, block_size=32)


SIMDGROUP_HEADER = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
"""


matmul_tile_simdgroup_kernel = mx.fast.metal_kernel(
    name="matmul_tile_simdgroup",
    input_names=["A", "B"],
    output_names=["C"],
    header=SIMDGROUP_HEADER,
    source=r"""
        uint block_x = threadgroup_position_in_grid.x;
        uint block_y = threadgroup_position_in_grid.y;

        uint simd_id = simdgroup_index_in_threadgroup;

        const uint TILE_DIM = 8;

        uint tg_tile_row_base = block_y * 32;
        uint tg_tile_col_base = block_x * 16;

        uint local_simd_row = (simd_id / 2) * TILE_DIM;
        uint local_simd_col = (simd_id % 2) * TILE_DIM;

        uint c_row = tg_tile_row_base + local_simd_row;
        uint c_col = tg_tile_col_base + local_simd_col;

        if (c_row >= M || c_col >= N) {
            return;
        }

        simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        for (uint k = 0; k < K; k += TILE_DIM) {
            device const float* a_ptr = A + c_row * K + k;
            device const float* b_ptr = B + k * N + c_col;

            simdgroup_float8x8 a_tile;
            simdgroup_float8x8 b_tile;

            simdgroup_load(a_tile, a_ptr, K);
            simdgroup_load(b_tile, b_ptr, N);
            simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
        }

        simdgroup_store(acc, C + c_row * N + c_col, N);
    """,
)


def matmul_tile_simdgroup(A: mx.array, B: mx.array) -> mx.array:
    M, K = A.shape
    _, N = B.shape

    if A.dtype != mx.float32 or B.dtype != mx.float32:
        raise ValueError("simdgroup GEMM currently supports float32 inputs only")
    if M % 8 or N % 8 or K % 8:
        raise ValueError("simdgroup GEMM requires M, N, and K to be divisible by 8")

    threadgroup_tile_m = 32
    threadgroup_tile_n = 16
    threadgroup_threads = 256
    grid = (
        ((N + threadgroup_tile_n - 1) // threadgroup_tile_n) * threadgroup_threads,
        (M + threadgroup_tile_m - 1) // threadgroup_tile_m,
        1,
    )

    (C,) = matmul_tile_simdgroup_kernel(
        inputs=[A, B],
        output_shapes=[(M, N)],
        output_dtypes=[A.dtype],
        template=[
            ("M", M),
            ("N", N),
            ("K", K),
        ],
        grid=grid,
        threadgroup=(threadgroup_threads, 1, 1),
    )
    return C


GEMM_KERNELS = {
    "mlx": mlx_matmul,
    "simt": matmul_naive_block_style,
    "tiled16": matmul_tiled_16,
    "tiled32": matmul_tiled_32,
    "simdgroup": matmul_tile_simdgroup,
}
