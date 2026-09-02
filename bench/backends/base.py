"""Benchmark backend protocol and shared result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class StageResult:
    label: str
    kind: str
    endpoint: str | None
    duration_ms: float
    price_usd: float
    ok: bool = True
    detail: str | None = None


@dataclass
class BenchmarkResult:
    fixture: str
    backend: str
    replay: bool
    image_url: str
    stages: list[StageResult] = field(default_factory=list)
    fal_calls: list[dict[str, Any]] = field(default_factory=list)
    api_cost_usd: float = 0.0
    total_ms: float = 0.0
    wall_height_cm: float = 260.0
    wall_height_source: str = "stub"

    def stage_labels(self) -> list[str]:
        return [s.label for s in self.stages]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture": self.fixture,
            "backend": self.backend,
            "replay": self.replay,
            "image_url": self.image_url,
            "stages": [
                {
                    "label": s.label,
                    "kind": s.kind,
                    "endpoint": s.endpoint,
                    "duration_ms": s.duration_ms,
                    "price_usd": s.price_usd,
                    "ok": s.ok,
                    "detail": s.detail,
                }
                for s in self.stages
            ],
            "fal_calls": self.fal_calls,
            "api_cost_usd": self.api_cost_usd,
            "total_ms": self.total_ms,
            "wall_height_cm": self.wall_height_cm,
            "wall_height_source": self.wall_height_source,
        }


class BenchmarkBackend(Protocol):
    def run(self, fixture_slug: str) -> BenchmarkResult: ...
