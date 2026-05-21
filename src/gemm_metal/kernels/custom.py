# ❤️  Written by Codex
"""Minimal helper for MLX custom Metal kernels.

This module intentionally does not provide a GEMM launcher. Each handwritten
kernel should choose its own input names, output names, templates, grid, and
threadgroup size next to the kernel body.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


@lru_cache(maxsize=None)
def metal_kernel(
    *,
    name: str,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    source: str,
    header: str = "",
    ensure_row_contiguous: bool = True,
    atomic_outputs: bool = False,
):
    """Thin cached wrapper around ``mx.fast.metal_kernel``.

    Kernel modules may also call ``mx.fast.metal_kernel`` directly. This helper
    only avoids recompiling/recreating the same kernel object repeatedly.
    """
    return mx.fast.metal_kernel(
        name=name,
        input_names=input_names,
        output_names=output_names,
        source=source,
        header=header,
        ensure_row_contiguous=ensure_row_contiguous,
        atomic_outputs=atomic_outputs,
    )
