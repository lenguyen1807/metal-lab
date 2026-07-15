from __future__ import annotations

import mlx.core as mx


def assert_close(
    actual: mx.array,
    expected: mx.array,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> None:
    mx.eval(actual, expected)
    if actual.shape != expected.shape:
        raise AssertionError(f"shape mismatch: actual={actual.shape}, expected={expected.shape}")
    if actual.dtype != expected.dtype:
        raise AssertionError(f"dtype mismatch: actual={actual.dtype}, expected={expected.dtype}")

    close = mx.all(mx.abs(actual - expected) <= atol + rtol * mx.abs(expected))
    mx.eval(close)
    if bool(close.item()):
        return

    max_error = mx.max(mx.abs(actual - expected))
    mean_error = mx.mean(mx.abs(actual - expected))
    mx.eval(max_error, mean_error)
    raise AssertionError(
        "arrays are not close: "
        f"max_abs_error={float(max_error.item()):.6e}, "
        f"mean_abs_error={float(mean_error.item()):.6e}, "
        f"rtol={rtol}, atol={atol}"
    )
