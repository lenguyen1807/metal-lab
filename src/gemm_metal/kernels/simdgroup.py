# ❤️  Written by Codex
"""SIMD-group GEMM kernel slot.

MLX generates the Metal function signature. Keep orchestration local: this file
should define the body string, create the custom kernel, choose template values,
choose grid/threadgroup sizes, and return the output.

This file is intentionally not implemented by the agent.
"""

from __future__ import annotations

import mlx.core as mx


# HANDWRITTEN KERNEL AREA START

SIMDGROUP_SOURCE = r"""
// Write the SIMD-group GEMM MLX custom Metal function body here by hand.
"""


def simdgroup(a: mx.array, b: mx.array) -> mx.array:
    """Handwritten SIMD-group custom Metal GEMM should live here."""
    raise NotImplementedError("Write the SIMD-group MLX custom Metal kernel body and launcher by hand here.")

# HANDWRITTEN KERNEL AREA END
