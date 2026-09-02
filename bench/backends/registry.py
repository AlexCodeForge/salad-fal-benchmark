"""Backend registry."""

from __future__ import annotations

from bench.backends.base import BenchmarkBackend
from bench.backends.fal import FalBackend
from bench.backends.salad import SaladBenchmarkBackend


def get_backend(name: str, *, replay: bool = False) -> BenchmarkBackend:
    if name == "fal":
        return FalBackend(replay=replay)
    if name == "salad":
        return SaladBenchmarkBackend()
    known = "fal, salad"
    raise ValueError(f"unknown backend {name!r}; known: {known}")
