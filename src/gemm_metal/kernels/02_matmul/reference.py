from __future__ import annotations

import mlx.core as mx

from gemm_metal.kernels.utils import check_matmul_inputs


def matmul_ref(A: mx.array, B: mx.array, M: int, N: int, K: int) -> mx.array:
    check_matmul_inputs(A, B, M, N, K)
    return A @ B
