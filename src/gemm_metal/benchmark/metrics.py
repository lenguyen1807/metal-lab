# ❤️  Written by Codex
"""GEMM performance metrics.

Two related conventions are useful:

- FLOP accounting usually counts one multiply-add as 2 floating-point ops.
- The Apple GEMM-performance text often discusses arithmetic intensity as
  multiply-add operations per matrix element moved.

This module exposes both so benchmark tables can report standard GFLOP/s while
roofline notes can also use the element-based formulas from the reference text.

Reference: Section 4.1 of https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf
"""

from __future__ import annotations


def fma_ops(m: int, n: int, k: int) -> int:
    """Multiply-add operations for C = A @ B."""
    return m * n * k


def flops(m: int, n: int, k: int) -> int:
    """Floating-point operations, counting one FMA as two FLOPs."""
    return 2 * fma_ops(m, n, k)


def input_elements(m: int, n: int, k: int) -> int:
    """Input elements read once in the ideal-reuse model: A plus B."""
    return m * k + n * k


def output_elements(m: int, n: int, k: int) -> int:
    """Output elements written once: C."""
    return m * n


def total_elements(m: int, n: int, k: int) -> int:
    """Ideal total elements moved: A, B, and C once."""
    return input_elements(m, n, k) + output_elements(m, n, k)


def fp32_minimum_bytes(m: int, n: int, k: int) -> int:
    """Ideal lower-bound FP32 traffic: read A/B once, write C once."""
    return 4 * total_elements(m, n, k)


def ops_per_input_element(m: int, n: int, k: int) -> float:
    """Reference formula: (M*N*K) / ((M*K) + (N*K)) == (M*N)/(M+N)."""
    return fma_ops(m, n, k) / input_elements(m, n, k)


def ops_per_output_element(m: int, n: int, k: int) -> float:
    """Reference formula: (M*N*K) / (M*N) == K."""
    return fma_ops(m, n, k) / output_elements(m, n, k)


def ops_per_total_element(m: int, n: int, k: int) -> float:
    """Reference total arithmetic intensity in FMA ops per ideal element moved."""
    return fma_ops(m, n, k) / total_elements(m, n, k)


def arithmetic_intensity(m: int, n: int, k: int) -> float:
    """FLOPs per byte using ideal FP32 traffic; suitable for roofline plots."""
    return flops(m, n, k) / fp32_minimum_bytes(m, n, k)


def gflops(m: int, n: int, k: int, seconds: float) -> float:
    """GFLOP/s for a GEMM runtime."""
    return flops(m, n, k) / seconds / 1e9


def bandwidth_gbs(m: int, n: int, k: int, seconds: float) -> float:
    """Effective GB/s using the ideal FP32 traffic estimate."""
    return fp32_minimum_bytes(m, n, k) / seconds / 1e9
