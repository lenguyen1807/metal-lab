from __future__ import annotations

import mlx.core as mx

SUPPORTED_DTYPES = (mx.float32,)


def check_matmul_inputs(A: mx.array, B: mx.array, M: int, N: int, K: int) -> None:
    check_mnk(M, N, K)
    check_array_2d("A", A)
    check_array_2d("B", B)
    check_dtype("A", A)
    check_dtype("B", B)

    if A.dtype != B.dtype:
        raise ValueError(f"A and B must have the same dtype: A={A.dtype}, B={B.dtype}")
    if A.shape != (M, K):
        raise ValueError(f"A shape must match (M, K): expected={(M, K)}, actual={A.shape}")
    if B.shape != (K, N):
        raise ValueError(f"B shape must match (K, N): expected={(K, N)}, actual={B.shape}")


def check_mnk(M: int, N: int, K: int) -> None:
    for name, value in (("M", M), ("N", N), ("K", K)):
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


def check_array_2d(name: str, value: mx.array) -> None:
    if not isinstance(value, mx.array):
        raise TypeError(f"{name} must be an mlx.core.array, got {type(value).__name__}")
    if value.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {value.shape}")


def check_dtype(name: str, value: mx.array) -> None:
    if value.dtype not in SUPPORTED_DTYPES:
        supported = ", ".join(str(dtype) for dtype in SUPPORTED_DTYPES)
        raise TypeError(f"{name} dtype must be one of ({supported}), got {value.dtype}")
