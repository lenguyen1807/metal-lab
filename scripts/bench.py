#!/usr/bin/env python3
# ❤️  Written by Codex
"""Convenience wrapper for running the full GEMM benchmark suite."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from gemm_metal.benchmark.cases import all_cases
from gemm_metal.benchmark.runner import BenchmarkConfig, benchmark_case, write_csv
from gemm_metal.kernels import get_kernel, kernel_names


def kernels_to_benchmark(target: str) -> list[str]:
    """Always include the production MLX kernel for comparison."""
    return ["mlx"] if target == "mlx" else ["mlx", target]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="mlx", choices=kernel_names())
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()

    config = BenchmarkConfig(
        warmups=args.warmups,
        iterations=args.iterations,
        seed=args.seed,
        validate=True,
    )

    results = []
    for kernel_name in kernels_to_benchmark(args.kernel):
        kernel = get_kernel(kernel_name)
        for case in all_cases():
            result = benchmark_case(
                kernel_name=kernel_name,
                kernel=kernel,
                case=case,
                config=config,
            )
            results.append(result)
            print(
                f"{result.kernel:>14} {result.case:<24} "
                f"median={result.median_ms:8.3f} ms "
                f"gflops={result.gflops:9.2f} "
                f"AI={result.arithmetic_intensity:7.3f} flop/byte"
            )

    if args.output:
        output = Path(args.output)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path("outputs/raw") / f"bench_{args.kernel}_with_mlx_{stamp}.csv"

    write_csv(results, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
