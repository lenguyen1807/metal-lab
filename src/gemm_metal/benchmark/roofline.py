# ❤️  Written by Codex
"""Small roofline-model helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RooflineMachine:
    name: str = "Apple M2 8-core GPU"
    peak_gflops: float = 2840.0
    peak_bandwidth_gbs: float = 100.0


def roofline_ceiling_gflops(arithmetic_intensity: float, machine: RooflineMachine) -> float:
    """Return roofline performance ceiling for a given arithmetic intensity."""
    return min(machine.peak_gflops, arithmetic_intensity * machine.peak_bandwidth_gbs)


def bound_type(arithmetic_intensity: float, machine: RooflineMachine) -> str:
    """Classify whether the roofline ceiling is memory or compute bound."""
    ridge_point = machine.peak_gflops / machine.peak_bandwidth_gbs
    return "memory" if arithmetic_intensity < ridge_point else "compute"
