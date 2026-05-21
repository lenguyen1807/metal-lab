# ❤️  Written by Codex
"""Benchmark runner."""

from __future__ import annotations

import csv
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import mlx.core as mx

from gemm_metal.benchmark.cases import GemmCase
from gemm_metal.benchmark.metrics import (
    arithmetic_intensity,
    bandwidth_gbs,
    gflops,
    ops_per_input_element,
    ops_per_output_element,
    ops_per_total_element,
)
from gemm_metal.reference import make_inputs, max_abs_error, mean_abs_error, mlx_matmul

KernelFn = Callable[[mx.array, mx.array], mx.array]


@dataclass(frozen=True)
class BenchmarkConfig:
    warmups: int = 5
    iterations: int = 20
    seed: int = 0
    validate: bool = True


@dataclass(frozen=True)
class BenchmarkResult:
    timestamp: str
    kernel: str
    case: str
    family: str
    m: int
    n: int
    k: int
    warmups: int
    iterations: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    gflops: float
    bandwidth_gbs: float
    arithmetic_intensity: float
    ops_per_input_element: float
    ops_per_output_element: float
    ops_per_total_element: float
    max_abs_error: float | None
    mean_abs_error: float | None
    python: str
    mlx: str


def benchmark_case(
    *,
    kernel_name: str,
    kernel: KernelFn,
    case: GemmCase,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Benchmark one kernel on one GEMM shape."""
    a, b = make_inputs(case.m, case.n, case.k, seed=config.seed)

    max_err: float | None = None
    mean_err: float | None = None
    if config.validate:
        expected = mlx_matmul(a, b)
        actual = kernel(a, b)
        mx.eval(expected, actual)
        max_err = max_abs_error(actual, expected)
        mean_err = mean_abs_error(actual, expected)

    for _ in range(config.warmups):
        out = kernel(a, b)
        mx.eval(out)

    times: list[float] = []
    for _ in range(config.iterations):
        start = time.perf_counter()
        out = kernel(a, b)
        mx.eval(out)
        end = time.perf_counter()
        times.append(end - start)

    mean_s = statistics.fmean(times)
    median_s = statistics.median(times)
    std_s = statistics.stdev(times) if len(times) > 1 else 0.0

    return BenchmarkResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        kernel=kernel_name,
        case=case.name,
        family=case.family,
        m=case.m,
        n=case.n,
        k=case.k,
        warmups=config.warmups,
        iterations=config.iterations,
        mean_ms=mean_s * 1e3,
        median_ms=median_s * 1e3,
        min_ms=min(times) * 1e3,
        max_ms=max(times) * 1e3,
        std_ms=std_s * 1e3,
        gflops=gflops(case.m, case.n, case.k, median_s),
        bandwidth_gbs=bandwidth_gbs(case.m, case.n, case.k, median_s),
        arithmetic_intensity=arithmetic_intensity(case.m, case.n, case.k),
        ops_per_input_element=ops_per_input_element(case.m, case.n, case.k),
        ops_per_output_element=ops_per_output_element(case.m, case.n, case.k),
        ops_per_total_element=ops_per_total_element(case.m, case.n, case.k),
        max_abs_error=max_err,
        mean_abs_error=mean_err,
        python=platform.python_version(),
        mlx=mx.__version__,
    )


def write_csv(results: list[BenchmarkResult], path: Path) -> None:
    """Write benchmark results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
