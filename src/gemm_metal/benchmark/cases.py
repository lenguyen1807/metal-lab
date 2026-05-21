# ❤️  Written by Codex
"""Benchmark shape catalog.

The benchmark intentionally runs every case by default. This keeps the CLI simple
and makes each benchmark output directly comparable across kernels.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GemmCase:
    name: str
    m: int
    n: int
    k: int
    family: str


CASES: list[GemmCase] = [
    # Small / smoke-test cases.
    GemmCase("small_square_128", 128, 128, 128, "small"),
    GemmCase("small_odd", 127, 131, 129, "small"),
    # Square powers of two.
    GemmCase("square_256", 256, 256, 256, "square"),
    GemmCase("square_512", 512, 512, 512, "square"),
    GemmCase("square_1024", 1024, 1024, 1024, "square"),
    GemmCase("square_2048", 2048, 2048, 2048, "square"),
    # Transformer-like shapes.
    GemmCase("ffn_1x512x2048", 512, 2048, 512, "ffn"),
    GemmCase("ffn_1x1024x4096", 1024, 4096, 1024, "ffn"),
    GemmCase("attention_qk_1024", 1024, 1024, 128, "attention"),
    GemmCase("attention_av_1024", 1024, 128, 1024, "attention"),
    # Skinny/tall cases.
    GemmCase("skinny_k16", 4096, 4096, 16, "skinny"),
    GemmCase("skinny_k32", 4096, 4096, 32, "skinny"),
    GemmCase("tall_m8192", 8192, 256, 256, "tall"),
    # Odd / non-ideal boundaries.
    GemmCase("odd_511_769_257", 511, 769, 257, "odd"),
    GemmCase("odd_1000_1001_1002", 1000, 1001, 1002, "odd"),
]


def all_cases() -> list[GemmCase]:
    """Return the full benchmark shape catalog."""
    return list(CASES)
