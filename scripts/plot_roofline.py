#!/usr/bin/env python3
# ❤️  Written by Codex
"""Generate a simple dependency-free SVG roofline plot in HTML."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", type=Path, help="CSV files; defaults to outputs/raw/*.csv")
    parser.add_argument("--output", type=Path, default=Path("outputs/plots/roofline.html"))
    parser.add_argument("--peak-gflops", type=float, default=2840.0)
    parser.add_argument("--peak-bandwidth-gbs", type=float, default=100.0)
    args = parser.parse_args()

    inputs = args.inputs or sorted(Path("outputs/raw").glob("*.csv"))
    rows: list[dict[str, str]] = []
    for path in inputs:
        with path.open(newline="") as f:
            rows.extend(csv.DictReader(f))

    points = [
        (float(row["arithmetic_intensity"]), float(row["gflops"]), row["kernel"], row["case"])
        for row in rows
        if float(row["arithmetic_intensity"]) > 0 and float(row["gflops"]) > 0
    ]

    width, height = 900, 560
    left, right, top, bottom = 80, 30, 30, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    max_ai = max([p[0] for p in points] + [args.peak_gflops / args.peak_bandwidth_gbs, 1.0]) * 2
    max_perf = max([p[1] for p in points] + [args.peak_gflops, 1.0]) * 1.1
    min_ai = 0.1
    min_perf = 1.0

    def sx(ai: float) -> float:
        return left + (math.log10(ai) - math.log10(min_ai)) / (math.log10(max_ai) - math.log10(min_ai)) * plot_w

    def sy(perf: float) -> float:
        return top + (math.log10(max(perf, min_perf)) - math.log10(max_perf)) / (math.log10(min_perf) - math.log10(max_perf)) * plot_h

    ridge = args.peak_gflops / args.peak_bandwidth_gbs
    roof_points = [
        (min_ai, min(args.peak_gflops, min_ai * args.peak_bandwidth_gbs)),
        (ridge, args.peak_gflops),
        (max_ai, args.peak_gflops),
    ]
    roof_polyline = " ".join(f"{sx(ai):.1f},{sy(perf):.1f}" for ai, perf in roof_points)

    circles = []
    for ai, perf, kernel, case in points:
        circles.append(
            f"<circle cx='{sx(ai):.1f}' cy='{sy(perf):.1f}' r='4'><title>{kernel} {case}: AI={ai:.3f}, {perf:.2f} GFLOP/s</title></circle>"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Roofline</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
svg {{ border: 1px solid #ddd; max-width: 100%; }}
.roof {{ fill: none; stroke: #dc2626; stroke-width: 3; }}
circle {{ fill: #4f46e5; opacity: .8; }}
.axis {{ stroke: #111; stroke-width: 1; }}
</style></head><body>
<h1>Roofline</h1>
<p>Peak compute: {args.peak_gflops:.0f} GFLOP/s. Peak bandwidth: {args.peak_bandwidth_gbs:.0f} GB/s.</p>
<svg viewBox='0 0 {width} {height}'>
<line class='axis' x1='{left}' y1='{height-bottom}' x2='{width-right}' y2='{height-bottom}' />
<line class='axis' x1='{left}' y1='{top}' x2='{left}' y2='{height-bottom}' />
<polyline class='roof' points='{roof_polyline}' />
{''.join(circles)}
<text x='{width/2:.0f}' y='{height-20}' text-anchor='middle'>Arithmetic intensity (FLOP/byte, log scale)</text>
<text x='20' y='{height/2:.0f}' transform='rotate(-90 20 {height/2:.0f})' text-anchor='middle'>GFLOP/s (log scale)</text>
</svg>
</body></html>\n"""
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
