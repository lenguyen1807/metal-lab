"""Vector addition."""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


def mlx_add(a: mx.array, b: mx.array) -> mx.array:
    return a + b


vadd_kernel = mx.fast.metal_kernel(
    name="vadd",
    input_names=["a", "b"],
    output_names=["out"],
    source=r"""
        uint i = thread_position_in_grid.x;
        out[i] = a[i] + b[i];
    """,
)


def metal_add(a: mx.array, b: mx.array) -> mx.array:
    (out,) = vadd_kernel(
        inputs=[a, b],
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
        grid=(a.size, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return out


VADD_KERNELS = {
    "mlx": mlx_add,
    "metal": metal_add,
}
