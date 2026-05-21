# ❤️  Written by Codex
"""Register-tiled GEMM kernel slot.

MLX generates the Metal function signature. Keep orchestration local: this file
should define the body string, create the custom kernel, choose template values,
choose grid/threadgroup sizes, and return the output.

This file is intentionally not implemented by the agent.
"""

from __future__ import annotations

import mlx.core as mx


# HANDWRITTEN KERNEL AREA START

REGISTER_TILED_SOURCE = r"""
// Write the register-tiled GEMM MLX custom Metal function body here by hand.
"""


def register_tiled(a: mx.array, b: mx.array) -> mx.array:
    """Handwritten register-tiled custom Metal GEMM should live here."""
    raise NotImplementedError("Write the register-tiled MLX custom Metal kernel body and launcher by hand here.")

# HANDWRITTEN KERNEL AREA END
