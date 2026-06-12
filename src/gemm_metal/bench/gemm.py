"""GEMM benchmark spec."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from gemm_metal.bench.common import BenchSpec
from gemm_metal.ops.gemm import GEMM_KERNELS, mlx_matmul


@dataclass(frozen=True)
class GemmCase:
    name: str
    M: int
    N: int
    K: int
    family: str


CASES = [
    GemmCase("square_256", 256, 256, 256, "square"),
    GemmCase("square_512", 512, 512, 512, "square"),
    GemmCase("square_768", 768, 768, 768, "square"),
    GemmCase("square_1024", 1024, 1024, 1024, "square"),
    GemmCase("square_1536", 1536, 1536, 1536, "square"),
    GemmCase("square_2048", 2048, 2048, 2048, "square"),
    GemmCase("ffn_512_2048_512", 512, 2048, 512, "ffn"),
    GemmCase("ffn_1024_4096_1024", 1024, 4096, 1024, "ffn"),
    GemmCase("ffn_2048_4096_1024", 2048, 4096, 1024, "ffn"),
    GemmCase("attention_qk_1024", 1024, 1024, 128, "attention"),
    GemmCase("attention_av_1024", 1024, 128, 1024, "attention"),
    GemmCase("attention_qk_2048", 2048, 2048, 128, "attention"),
    GemmCase("attention_av_2048", 2048, 128, 2048, "attention"),
    GemmCase("skinny_k16", 1024, 1024, 16, "skinny"),
    GemmCase("skinny_k32", 2048, 2048, 32, "skinny"),
    GemmCase("tall_m4096", 4096, 256, 256, "tall"),
    GemmCase("wide_n4096", 256, 4096, 256, "wide"),
    GemmCase("odd_127_131_129", 127, 131, 129, "edge"),
    GemmCase("odd_255_257_129", 255, 257, 129, "edge"),
    GemmCase("odd_511_769_257", 511, 769, 257, "edge"),
    GemmCase("stress_attention_qk_4096", 4096, 4096, 128, "stress"),
    GemmCase("stress_attention_av_4096", 4096, 128, 4096, "stress"),
    GemmCase("stress_square_3072", 3072, 3072, 3072, "stress"),
    GemmCase("stress_ffn_4096_8192_4096", 4096, 8192, 4096, "stress"),
    GemmCase("extreme_square_4096", 4096, 4096, 4096, "extreme"),
    GemmCase("extreme_ffn_4096_11008_4096", 4096, 11008, 4096, "extreme"),
]


def cases(suite: str) -> list[GemmCase]:
    if suite == "square":
        return [case for case in CASES if case.family == "square"]
    if suite == "stress":
        return [case for case in CASES if case.family == "stress"]
    if suite == "extreme":
        return [case for case in CASES if case.family == "extreme"]
    if suite == "full":
        return [case for case in CASES if case.family not in {"stress", "extreme"}]
    raise ValueError(f"unknown suite: {suite}")


def inputs(case: GemmCase, seed: int) -> tuple[mx.array, mx.array]:
    mx.random.seed(seed)
    A = mx.random.normal((case.M, case.K), dtype=mx.float32)
    B = mx.random.normal((case.K, case.N), dtype=mx.float32)
    mx.eval(A, B)
    return A, B


SPEC = BenchSpec(
    op="gemm",
    kernels=GEMM_KERNELS,
    default_kernel="simt",
    cases=cases,
    inputs=inputs,
    baseline=mlx_matmul,
    flops=lambda case: float(2 * case.M * case.N * case.K),
    bytes=lambda case: float((case.M * case.K + case.K * case.N + case.M * case.N) * 4),
    shape=lambda case: f"{case.M}x{case.N}x{case.K}",
)
