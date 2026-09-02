"""Run-level report models for benchmark aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bench.backends.base import BenchmarkResult


@dataclass
class RunRecord:
    """One benchmark run with flat metric rows for CSV output."""

    run_id: int
    run_temperature: str
    result: BenchmarkResult
    rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RunSummary:
    """Aggregated headline metrics across N runs."""

    fixture: str
    backend: str
    runs: int
    latency: dict[str, float | int]
    cost: dict[str, float]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture": self.fixture,
            "backend": self.backend,
            "runs": self.runs,
            "latency": self.latency,
            "cost": self.cost,
            "meta": self.meta,
        }
