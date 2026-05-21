#!/usr/bin/env python3
# ❤️  Written by Codex
"""Generate a simple dependency-free HTML bar chart for kernel comparison."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", type=Path, help="CSV files; defaults to outputs/raw/*.csv")
    parser.add_argument("--output", type=Path, default=Path("outputs/plots/kernel_comparison.html"))
    args = parser.parse_args()

    inputs = args.inputs or sorted(Path("outputs/raw").glob("*.csv"))
    rows: list[dict[str, str]] = []
    for path in inputs:
        with path.open(newline="") as f:
            rows.extend(csv.DictReader(f))

    best: dict[str, float] = {}
    for row in rows:
        key = row["kernel"]
        best[key] = max(best.get(key, 0.0), float(row["gflops"]))

    max_value = max(best.values(), default=1.0)
    bars = []
    for kernel, value in sorted(best.items(), key=lambda item: item[1], reverse=True):
        width = 100 * value / max_value
        bars.append(
            f"<div class='row'><span class='label'>{html.escape(kernel)}</span>"
            f"<div class='bar' style='width:{width:.2f}%'></div>"
            f"<span class='value'>{value:.2f} GFLOP/s</span></div>"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        """<!doctype html>
<html><head><meta charset='utf-8'><title>Kernel comparison</title>
<style>
body { font-family: system-ui, sans-serif; margin: 2rem; }
.row { display: flex; align-items: center; gap: .75rem; margin: .5rem 0; }
.label { width: 10rem; text-align: right; font-family: monospace; }
.bar { height: 1.2rem; background: #4f46e5; min-width: 1px; }
.value { min-width: 10rem; }
</style></head><body>
<h1>Kernel comparison</h1>
<p>Best observed GFLOP/s per kernel.</p>
"""
        + "\n".join(bars)
        + "\n</body></html>\n"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
