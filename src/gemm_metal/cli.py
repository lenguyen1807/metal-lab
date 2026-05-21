# ❤️  Written by Codex
"""Command-line interface."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import mlx.core as mx

from gemm_metal.benchmark.cases import all_cases
from gemm_metal.benchmark.runner import BenchmarkConfig, BenchmarkResult, benchmark_case, write_csv
from gemm_metal.kernels import get_kernel, kernel_names
from gemm_metal.reference import make_inputs, max_abs_error, mean_abs_error, mlx_matmul


def _validate(args: argparse.Namespace) -> int:
    kernel = get_kernel(args.kernel)
    a, b = make_inputs(args.M, args.N, args.K, seed=args.seed)
    expected = mlx_matmul(a, b)
    actual = kernel(a, b)
    mx.eval(expected, actual)
    print(f"kernel={args.kernel} shape=({args.M}, {args.N}, {args.K})")
    print(f"max_abs_error={max_abs_error(actual, expected):.6g}")
    print(f"mean_abs_error={mean_abs_error(actual, expected):.6g}")
    return 0


def _kernels_to_benchmark(target: str) -> list[str]:
    """Always include the production MLX kernel for comparison."""
    return ["mlx"] if target == "mlx" else ["mlx", target]


def _print_result(result: BenchmarkResult) -> None:
    print(
        f"{result.kernel:>14} {result.case:<24} "
        f"median={result.median_ms:8.3f} ms "
        f"gflops={result.gflops:9.2f} "
        f"AI={result.arithmetic_intensity:7.3f} flop/byte"
    )


def _bench(args: argparse.Namespace) -> int:
    config = BenchmarkConfig(
        warmups=args.warmups,
        iterations=args.iterations,
        seed=args.seed,
        validate=True,
    )

    results: list[BenchmarkResult] = []
    for kernel_name in _kernels_to_benchmark(args.kernel):
        kernel = get_kernel(kernel_name)
        for case in all_cases():
            result = benchmark_case(
                kernel_name=kernel_name,
                kernel=kernel,
                case=case,
                config=config,
            )
            results.append(result)
            _print_result(result)

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("outputs/raw") / f"bench_{args.kernel}_with_mlx_{stamp}.csv"
    else:
        path = Path(args.output)
    write_csv(results, path)
    print(f"wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gemm-metal")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-kernels", help="List registered kernels")

    validate = sub.add_parser("validate", help="Validate one kernel against mlx.matmul")
    validate.add_argument("--kernel", default="mlx", choices=kernel_names())
    validate.add_argument("--M", type=int, default=128)
    validate.add_argument("--N", type=int, default=128)
    validate.add_argument("--K", type=int, default=128)
    validate.add_argument("--seed", type=int, default=0)

    bench = sub.add_parser("bench", help="Benchmark all cases; always includes mlx.matmul")
    bench.add_argument("--kernel", default="mlx", choices=kernel_names())
    bench.add_argument("--warmups", type=int, default=5)
    bench.add_argument("--iterations", type=int, default=20)
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--output")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list-kernels":
        for name in kernel_names():
            print(name)
        return
    if args.command == "validate":
        raise SystemExit(_validate(args))
    if args.command == "bench":
        raise SystemExit(_bench(args))

    parser.error(f"unknown command: {args.command}")
