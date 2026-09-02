"""Pipeline stage graph — re-exports stages.yaml loader for harness use."""

from __future__ import annotations

from bench.models.stage import StageDef, StageGraph, load_stage_graph


def load_pipeline_graph() -> StageGraph:
    """Load the Rust-prod sequential stage DAG."""
    return load_stage_graph()


__all__ = ["StageDef", "StageGraph", "load_pipeline_graph"]
