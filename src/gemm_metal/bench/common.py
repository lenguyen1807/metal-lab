"""Shared benchmark machinery."""

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

Kernel = Callable[..., mx.array]


@dataclass(frozen=True)
class Result:
    timestamp: str
    op: str
    kernel: str
    case: str
    family: str
    shape: str
    warmups: int
    iterations: int
    mean_us: float
    median_us: float
    min_us: float
    max_us: float
    std_us: float
    gflops: float | None
    bandwidth_gbs: float | None
    arithmetic_intensity: float | None
    max_abs_error: float
    mean_abs_error: float
    python: str
    mlx: str


@dataclass(frozen=True)
class BenchSpec:
    op: str
    kernels: dict[str, Kernel]
    default_kernel: str
    cases: Callable[[str], list[object]]
    inputs: Callable[[object, int], tuple[mx.array, ...]]
    baseline: Callable[..., mx.array]
    flops: Callable[[object], float | None]
    bytes: Callable[[object], float | None]
    shape: Callable[[object], str]


def max_abs_error(actual: mx.array, expected: mx.array) -> float:
    err = mx.max(mx.abs(actual - expected))
    mx.eval(err)
    return float(err.item())


def mean_abs_error(actual: mx.array, expected: mx.array) -> float:
    err = mx.mean(mx.abs(actual - expected))
    mx.eval(err)
    return float(err.item())


def selected(name: str) -> list[str]:
    return ["mlx"] if name == "mlx" else ["mlx", name]


def benchmark(
    *,
    spec: BenchSpec,
    kernel_name: str,
    case: object,
    warmups: int,
    iterations: int,
    seed: int,
) -> Result:
    args = spec.inputs(case, seed)
    expected = spec.baseline(*args)
    actual = spec.kernels[kernel_name](*args)
    mx.eval(expected, actual)

    max_err = max_abs_error(actual, expected)
    mean_err = mean_abs_error(actual, expected)

    for _ in range(warmups):
        mx.eval(spec.kernels[kernel_name](*args))

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        mx.eval(spec.kernels[kernel_name](*args))
        times.append(time.perf_counter() - start)

    mean_s = statistics.fmean(times)
    median_s = statistics.median(times)
    std_s = statistics.stdev(times) if len(times) > 1 else 0.0
    flops = spec.flops(case)
    traffic = spec.bytes(case)

    return Result(
        timestamp=datetime.now(timezone.utc).isoformat(),
        op=spec.op,
        kernel=kernel_name,
        case=getattr(case, "name"),
        family=getattr(case, "family"),
        shape=spec.shape(case),
        warmups=warmups,
        iterations=iterations,
        mean_us=mean_s * 1e6,
        median_us=median_s * 1e6,
        min_us=min(times) * 1e6,
        max_us=max(times) * 1e6,
        std_us=std_s * 1e6,
        gflops=None if flops is None else flops / median_s / 1e9,
        bandwidth_gbs=None if traffic is None else traffic / median_s / 1e9,
        arithmetic_intensity=None if flops is None or traffic is None else flops / traffic,
        max_abs_error=max_err,
        mean_abs_error=mean_err,
        python=platform.python_version(),
        mlx=mx.__version__,
    )


def write_csv(results: list[Result], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_result(result: Result) -> None:
    gflops = "" if result.gflops is None else f" {result.gflops:9.2f} GF/s"
    bandwidth = "" if result.bandwidth_gbs is None else f" {result.bandwidth_gbs:8.2f} GB/s"
    ai = "" if result.arithmetic_intensity is None else f" AI={result.arithmetic_intensity:7.3f}"
    print(
        f"{result.op:<5} {result.kernel:>5} {result.case:<18} {result.shape:<18} "
        f"median={result.median_us:9.2f} us{gflops}{bandwidth}{ai} "
        f"err={result.max_abs_error:.1e}"
    )

