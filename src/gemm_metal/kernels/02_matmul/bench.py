from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
from naive import matmul_naive
from reference import matmul_ref
from tiled import matmul_tiled_16, matmul_tiled_32

from gemm_metal.harness import (
    BenchResult,
    assert_close,
    benchmark_us,
    format_markdown_table,
    gbps,
    gflops,
)

Kernel = Callable[[mx.array, mx.array, int, int, int], mx.array]

KERNELS: dict[str, Kernel] = {
    "mlx": matmul_ref,
    "naive": matmul_naive,
    "tiled_16": matmul_tiled_16,
    "tiled_32": matmul_tiled_32,
}


@dataclass(frozen=True)
class MatmulCase:
    name: str
    M: int
    N: int
    K: int


CASES = [
    MatmulCase("square_256", 256, 256, 256),
    MatmulCase("square_512", 512, 512, 512),
    MatmulCase("square_1024", 1024, 1024, 1024),
    MatmulCase("odd_255_257_129", 255, 257, 129),
]


def inputs(case: MatmulCase, seed: int) -> tuple[mx.array, mx.array]:
    mx.random.seed(seed)
    A = mx.random.normal((case.M, case.K), dtype=mx.float32)
    B = mx.random.normal((case.K, case.N), dtype=mx.float32)
    mx.eval(A, B)
    return A, B


def kernel_flops(case: MatmulCase) -> float:
    return float(2 * case.M * case.N * case.K)


def bytes_moved(case: MatmulCase) -> float:
    return float((case.M * case.K + case.K * case.N + case.M * case.N) * 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MLX/Metal matmul kernels.")
    parser.add_argument("--kernel", choices=sorted(KERNELS), default="naive")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected = ["mlx"] if args.kernel == "mlx" else ["mlx", args.kernel]
    rows: list[BenchResult] = []

    for case in CASES:
        A, B = inputs(case, args.seed)
        expected = matmul_ref(A, B, case.M, case.N, case.K)

        for name in selected:
            fn = KERNELS[name]
            actual = fn(A, B, case.M, case.N, case.K)
            assert_close(actual, expected, rtol=1e-4, atol=1e-3)

            latency = benchmark_us(
                fn,
                A,
                B,
                case.M,
                case.N,
                case.K,
                warmup=args.warmup,
                rep=args.rep,
            )
            rows.append(
                BenchResult(
                    f"{case.name}/{name}",
                    latency,
                    gflops=gflops(kernel_flops(case), latency),
                    bandwidth_gbs=gbps(bytes_moved(case), latency),
                    notes=f"{case.M}x{case.N}x{case.K}",
                )
            )

    print(format_markdown_table(rows))
