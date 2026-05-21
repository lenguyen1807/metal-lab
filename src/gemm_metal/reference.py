# ❤️ Written by Codex
"""Reference GEMM implementations."""

from __future__ import annotations

import mlx.core as mx


def mlx_matmul(a: mx.array, b: mx.array) -> mx.array:
    """Reference MLX matmul baseline."""
    return mx.matmul(a, b)


def make_inputs(m: int, n: int, k: int, *, seed: int = 0) -> tuple[mx.array, mx.array]:
    """Create deterministic FP32 input matrices for C = A @ B."""
    mx.random.seed(seed)
    a = mx.random.normal((m, k)).astype(mx.float32)
    b = mx.random.normal((k, n)).astype(mx.float32)
    mx.eval(a, b)
    return a, b


def max_abs_error(actual: mx.array, expected: mx.array) -> float:
    """Return max absolute error between two MLX arrays."""
    err = mx.max(mx.abs(actual - expected))
    mx.eval(err)
    return float(err.item())


def mean_abs_error(actual: mx.array, expected: mx.array) -> float:
    """Return mean absolute error between two MLX arrays."""
    err = mx.mean(mx.abs(actual - expected))
    mx.eval(err)
    return float(err.item())
