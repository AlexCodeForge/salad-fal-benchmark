"""Sequential (prod) and optional parallel pipeline executor."""

from __future__ import annotations

from typing import Literal

from bench.backends.base import BenchmarkResult
from bench.backends.registry import get_backend
from bench.pipeline.graph import load_pipeline_graph
from bench.pipeline.postprocess import apply_postprocess

StageMode = Literal["sequential", "parallel"]


class PipelineExecutor:
    """Run Rust-prod analyze preprocess for one fixture via a registered backend."""

    def __init__(
        self,
        backend: str,
        *,
        replay: bool = False,
        stage_mode: StageMode = "sequential",
    ) -> None:
        if stage_mode not in {"sequential", "parallel"}:
            raise ValueError(f"unsupported stage_mode: {stage_mode!r}")
        if stage_mode == "parallel":
            raise NotImplementedError("parallel stage_mode is not yet implemented")
        self.backend_name = backend
        self.replay = replay
        self.stage_mode = stage_mode
        self.graph = load_pipeline_graph()
        if self.graph.mode != "sequential" and stage_mode == "sequential":
            pass  # graph SSOT; executor honors CLI stage_mode for future parallel

    def run(self, fixture_slug: str) -> BenchmarkResult:
        backend = get_backend(self.backend_name, replay=self.replay)
        result = backend.run(fixture_slug)
        post = apply_postprocess()
        result.wall_height_cm = post.wall_height_cm
        result.wall_height_source = post.wall_height_source
        return result
