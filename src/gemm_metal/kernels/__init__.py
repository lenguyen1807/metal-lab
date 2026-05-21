# ❤️  Written by Codex
"""Kernel registry."""

from __future__ import annotations

from typing import Callable

import mlx.core as mx

from gemm_metal.kernels.naive import naive
from gemm_metal.kernels.register_tiled import register_tiled
from gemm_metal.kernels.simdgroup import simdgroup
from gemm_metal.kernels.tiled import tiled
from gemm_metal.reference import mlx_matmul

KernelFn = Callable[[mx.array, mx.array], mx.array]


KERNELS: dict[str, KernelFn] = {
    "mlx": mlx_matmul,
    "naive": naive,
    "tiled": tiled,
    "register_tiled": register_tiled,
    "simdgroup": simdgroup,
}


def get_kernel(name: str) -> KernelFn:
    try:
        return KERNELS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(KERNELS))
        raise ValueError(f"Unknown kernel {name!r}. Valid kernels: {valid}") from exc


def kernel_names() -> list[str]:
    return sorted(KERNELS)
