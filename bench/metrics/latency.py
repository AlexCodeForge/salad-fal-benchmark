"""Latency aggregation: percentiles and cold/warm run tagging."""

from __future__ import annotations

from typing import Literal, Sequence

RunTemperature = Literal["cold", "warm"]


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100). Empty input returns 0.0."""
    if not values:
        return 0.0
    if p <= 0:
        return float(min(values))
    if p >= 100:
        return float(max(values))

    ordered = sorted(float(v) for v in values)
    rank = max(1, int(round(p / 100.0 * len(ordered))))
    return ordered[min(rank - 1, len(ordered) - 1)]


def summarize_latency(
    rows: Sequence[dict],
    *,
    field: str = "client_e2e_ms",
) -> dict[str, float | int]:
    """Aggregate L1-style latency from stage/run rows."""
    values = [float(row[field]) for row in rows if field in row and row[field] is not None]
    return {
        "n": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
    }


def tag_run_temperature(
    run_index: int,
    *,
    idle_minutes: float | None = None,
    container_age_minutes: float | None = None,
    backend: str = "fal",
) -> RunTemperature:
    """Classify a run as cold or warm per benchmark protocol."""
    if run_index <= 0:
        if backend == "salad" and container_age_minutes is not None:
            return "cold" if container_age_minutes < 5.0 else "warm"
        if idle_minutes is not None and idle_minutes >= 15.0:
            return "cold"
        return "cold"

    if backend == "salad":
        if container_age_minutes is not None and container_age_minutes < 5.0:
            return "cold"
        return "warm"

    if idle_minutes is not None and idle_minutes >= 15.0:
        return "cold"
    return "warm"
