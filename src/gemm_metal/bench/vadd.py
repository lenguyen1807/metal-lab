"""Vector-add benchmark spec."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from gemm_metal.bench.common import BenchSpec
from gemm_metal.ops.vadd import VADD_KERNELS, mlx_add


@dataclass(frozen=True)
class VaddCase:
    name: str
    n: int
    family: str


CASES = [
    VaddCase("1k", 1 << 10, "small"),
    VaddCase("64k", 1 << 16, "small"),
    VaddCase("1m", 1 << 20, "large"),
    VaddCase("16m", 1 << 24, "large"),
    VaddCase("odd_1009", 1009, "edge"),
    VaddCase("odd_1m_plus_3", (1 << 20) + 3, "edge"),
]


def cases(suite: str) -> list[VaddCase]:
    if suite == "square":
        return CASES[:2]
    if suite in {"full", "stress", "extreme"}:
        return list(CASES)
    raise ValueError(f"unknown suite: {suite}")


def inputs(case: VaddCase, seed: int) -> tuple[mx.array, mx.array]:
    mx.random.seed(seed)
    a = mx.random.normal((case.n,), dtype=mx.float32)
    b = mx.random.normal((case.n,), dtype=mx.float32)
    mx.eval(a, b)
    return a, b


SPEC = BenchSpec(
    op="vadd",
    kernels=VADD_KERNELS,
    default_kernel="metal",
    cases=cases,
    inputs=inputs,
    baseline=mlx_add,
    flops=lambda case: float(case.n),
    bytes=lambda case: float(case.n * 3 * 4),
    shape=lambda case: f"n={case.n}",
)
