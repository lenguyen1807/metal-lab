from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import mlx.core as mx


@dataclass(frozen=True)
class BenchResult:
    name: str
    latency_us: float
    gflops: float | None = None
    bandwidth_gbs: float | None = None
    notes: str = ""

    def as_row(self) -> list[str]:
        return [
            self.name,
            f"{self.latency_us:.2f}",
            "" if self.gflops is None else f"{self.gflops:.2f}",
            "" if self.bandwidth_gbs is None else f"{self.bandwidth_gbs:.2f}",
            self.notes,
        ]


def benchmark_us(
    fn: Callable[..., Any],
    *args: Any,
    warmup: int = 5,
    rep: int = 20,
    **kwargs: Any,
) -> float:
    for _ in range(warmup):
        mx.eval(fn(*args, **kwargs))

    samples: list[float] = []
    for _ in range(rep):
        start = time.perf_counter()
        mx.eval(fn(*args, **kwargs))
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples)


def gbps(bytes_moved: float, latency_us: float) -> float:
    return bytes_moved / latency_us / 1e3


def gflops(flops: float, latency_us: float) -> float:
    return flops / latency_us / 1e3


def format_markdown_table(
    rows: Iterable[BenchResult | Sequence[object]],
    headers: Sequence[str] = ("Kernel", "Latency (us)", "GFLOP/s", "GB/s", "Notes"),
) -> str:
    normalized_rows: list[list[str]] = []
    for row in rows:
        if isinstance(row, BenchResult):
            normalized_rows.append(row.as_row())
        else:
            normalized_rows.append(["" if value is None else str(value) for value in row])

    widths = [len(header) for header in headers]
    for row in normalized_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def fmt(values: Sequence[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        ) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt(list(headers)), separator, *(fmt(row) for row in normalized_rows)])
