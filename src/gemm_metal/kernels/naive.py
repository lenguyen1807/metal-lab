# ❤️  Written by Codex
"""Naive GEMM kernel slot.

MLX generates the Metal function signature. Keep orchestration local: this file
should define the body string, create the custom kernel, choose template values,
choose grid/threadgroup sizes, and return the output.

This file is intentionally not implemented by the agent.
"""

from __future__ import annotations

import mlx.core as mx


# HANDWRITTEN KERNEL AREA START

NAIVE_SOURCE = r"""
// Write the naive GEMM MLX custom Metal function body here by hand.
// Example choices left to you:
//   - input_names / output_names
//   - template values such as T, M, N, K, tile sizes
//   - grid and threadgroup shape
//   - output shape and dtype
"""


def naive(a: mx.array, b: mx.array) -> mx.array:
    """Handwritten naive custom Metal GEMM should live here."""
    raise NotImplementedError("Write the naive MLX custom Metal kernel body and launcher by hand here.")

# HANDWRITTEN KERNEL AREA END
