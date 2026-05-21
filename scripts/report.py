#!/usr/bin/env python3
# ❤️  Written by Codex
"""Generate a small Markdown report from benchmark CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                row["source_file"] = str(path)
                rows.append(row)
    return rows


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", type=Path, help="CSV files; defaults to outputs/raw/*.csv")
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/latest.md"))
    parser.add_argument("--peak-gflops", type=float, default=2840.0)
    args = parser.parse_args()

    inputs = args.inputs or sorted(Path("outputs/raw").glob("*.csv"))
    rows = read_rows(inputs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# GEMM benchmark report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Input CSV files: {len(inputs)}")
    lines.append(f"Rows: {len(rows)}")
    lines.append("")

    if not rows:
        lines.append("No benchmark rows found.")
        args.output.write_text("\n".join(lines) + "\n")
        print(f"wrote {args.output}")
        return

    best_by_kernel: dict[str, dict[str, str]] = {}
    for row in rows:
        kernel = row["kernel"]
        if kernel not in best_by_kernel or f(row, "gflops") > f(best_by_kernel[kernel], "gflops"):
            best_by_kernel[kernel] = row

    lines.append("## Best result per kernel")
    lines.append("")
    lines.append("| Kernel | Case | M | N | K | Median ms | GFLOP/s | % peak |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for kernel, row in sorted(best_by_kernel.items()):
        pct = f(row, "gflops") / args.peak_gflops * 100
        lines.append(
            f"| `{kernel}` | {row['case']} | {row['m']} | {row['n']} | {row['k']} | "
            f"{f(row, 'median_ms'):.3f} | {f(row, 'gflops'):.2f} | {pct:.2f}% |"
        )
    lines.append("")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["kernel"], row["family"])].append(row)

    lines.append("## Mean GFLOP/s by kernel and family")
    lines.append("")
    lines.append("| Kernel | Family | Mean GFLOP/s | Cases |")
    lines.append("|---|---:|---:|---:|")
    for (kernel, family), group in sorted(grouped.items()):
        mean = sum(f(row, "gflops") for row in group) / len(group)
        lines.append(f"| `{kernel}` | {family} | {mean:.2f} | {len(group)} |")
    lines.append("")

    baseline_by_file_case = {
        (row["source_file"], row["case"]): row
        for row in rows
        if row["kernel"] == "mlx"
    }
    comparisons = []
    for row in rows:
        if row["kernel"] == "mlx":
            continue
        baseline = baseline_by_file_case.get((row["source_file"], row["case"]))
        if baseline is None:
            continue
        comparisons.append((row, baseline))

    if comparisons:
        lines.append("## Comparison with production `mlx.matmul`")
        lines.append("")
        lines.append("| Kernel | Case | Kernel GFLOP/s | MLX GFLOP/s | Speedup vs MLX |")
        lines.append("|---|---:|---:|---:|---:|")
        for row, baseline in comparisons:
            speedup = f(row, "gflops") / f(baseline, "gflops")
            lines.append(
                f"| `{row['kernel']}` | {row['case']} | {f(row, 'gflops'):.2f} | "
                f"{f(baseline, 'gflops'):.2f} | {speedup:.3f}x |"
            )
        lines.append("")

    lines.append("## Correctness summary")
    lines.append("")
    lines.append("| Kernel | Max observed abs error |")
    lines.append("|---|---:|")
    for kernel in sorted({row["kernel"] for row in rows}):
        errs = [f(row, "max_abs_error") for row in rows if row["kernel"] == kernel and row.get("max_abs_error")]
        max_err = max(errs) if errs else 0.0
        lines.append(f"| `{kernel}` | {max_err:.6g} |")

    args.output.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
