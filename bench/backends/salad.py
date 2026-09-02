"""Salad GPU backend — same sequential SAM3 + depth pipeline as fal, via gateways."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin

from bench.backends.base import BenchmarkResult, StageResult as BaseStageResult
from bench.config import BenchSettings, get_settings
from bench.fixtures.loader import resolve_fixture
from bench.salad.client import SaladGatewayClient
from bench.salad.upload import build_depth_post, build_sam3_post, image_dimensions

SAM3_PATH = "/v1/sam3"
DEPTH_PATH = "/v1/depth"

WALL_SAM3_STAGES: tuple[tuple[str, str, str, int], ...] = (
    ("wall:wall:sam3", "wall", "wall", 8),
    ("wall:molding:sam3", "wall", "molding", 4),
    ("wall:mullion:sam3", "wall", "mullion", 4),
)
FLOOR_SAM3_STAGE = ("floor:floor:sam3", "floor", "floor", 8)
DEPTH_LABEL = "depth"

WALL_HEIGHT_CM_STUB = 260.0
WALL_HEIGHT_SOURCE_STUB = "stub"


@dataclass
class StageResult:
    label: str
    latency_ms: float
    success: bool
    response: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class SaladSegmentOutput:
    stages: list[StageResult] = field(default_factory=list)
    total_ms: float = 0.0
    image_w: int = 0
    image_h: int = 0
    wall_height_cm: float = WALL_HEIGHT_CM_STUB
    wall_height_source: str = WALL_HEIGHT_SOURCE_STUB
    upload_ms: float = 0.0


def _gateway_url(base: str, path: str) -> str:
    return urljoin(base.rstrip("/") + "/", path.lstrip("/"))


class SaladBackend:
    """Run Rust-prod segment_fal scope against Salad SAM3 + depth gateways."""

    def __init__(
        self,
        settings: BenchSettings | None = None,
        *,
        client: SaladGatewayClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._owns_client = client is None
        self._client = client or SaladGatewayClient(
            api_key=self.settings.salad_api_key,
            timeout=self.settings.bench_http_timeout_s,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> SaladBackend:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def segment(self, image_bytes: bytes) -> SaladSegmentOutput:
        """Sequential 4× SAM3 + depth; VLM stubbed at 260 cm."""
        sam3_url = self.settings.salad_sam3_gateway_url
        depth_url = self.settings.salad_depth_gateway_url
        if not sam3_url or not depth_url:
            raise ValueError(
                "SALAD_SAM3_GATEWAY_URL and SALAD_DEPTH_GATEWAY_URL must be set"
            )

        image_w, image_h = image_dimensions(image_bytes)
        stages: list[StageResult] = []
        pipeline_start = time.perf_counter()

        upload_start = time.perf_counter()
        upload_ms = (time.perf_counter() - upload_start) * 1000.0

        for label, _class_name, prompt, max_masks in (*WALL_SAM3_STAGES, FLOOR_SAM3_STAGE):
            stage = self._run_sam3(
                _gateway_url(sam3_url, SAM3_PATH),
                image_bytes,
                label=label,
                prompt=prompt,
                max_masks=max_masks,
            )
            stages.append(stage)

        stages.append(
            self._run_depth(_gateway_url(depth_url, DEPTH_PATH), image_bytes)
        )

        total_ms = (time.perf_counter() - pipeline_start) * 1000.0
        return SaladSegmentOutput(
            stages=stages,
            total_ms=total_ms,
            image_w=image_w,
            image_h=image_h,
            upload_ms=upload_ms,
        )

    def _run_sam3(
        self,
        url: str,
        image_bytes: bytes,
        *,
        label: str,
        prompt: str,
        max_masks: int,
    ) -> StageResult:
        start = time.perf_counter()
        try:
            payload = build_sam3_post(image_bytes, prompt, max_masks)
            assert isinstance(payload, tuple)
            files, form = payload
            response = self._client.post_multipart(url, data=form, files=files)
            latency_ms = (time.perf_counter() - start) * 1000.0
            return StageResult(
                label=label,
                latency_ms=latency_ms,
                success=True,
                response=response,
            )
        except Exception as exc:  # noqa: BLE001 — stage boundary; preserve pipeline errors
            latency_ms = (time.perf_counter() - start) * 1000.0
            return StageResult(
                label=label,
                latency_ms=latency_ms,
                success=False,
                error=str(exc),
            )

    def _run_depth(self, url: str, image_bytes: bytes) -> StageResult:
        start = time.perf_counter()
        try:
            payload = build_depth_post(image_bytes)
            assert isinstance(payload, tuple)
            files, form = payload
            response = self._client.post_multipart(url, data=form, files=files)
            latency_ms = (time.perf_counter() - start) * 1000.0
            return StageResult(
                label=DEPTH_LABEL,
                latency_ms=latency_ms,
                success=True,
                response=response,
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - start) * 1000.0
            return StageResult(
                label=DEPTH_LABEL,
                latency_ms=latency_ms,
                success=False,
                error=str(exc),
            )


def segment_salad(
    image_bytes: bytes,
    settings: BenchSettings | None = None,
    *,
    client: SaladGatewayClient | None = None,
) -> SaladSegmentOutput:
    """Convenience wrapper around SaladBackend.segment."""
    with SaladBackend(settings=settings, client=client) as backend:
        return backend.segment(image_bytes)


def _salad_stage_kind(label: str) -> str:
    if label == "depth":
        return "depth"
    if label.endswith(":sam3"):
        return "sam3"
    return "unknown"


class SaladBenchmarkBackend:
    """BenchmarkBackend adapter for Salad gateways."""

    def __init__(
        self,
        settings: BenchSettings | None = None,
        *,
        client: SaladGatewayClient | None = None,
    ) -> None:
        self._backend = SaladBackend(settings=settings, client=client)

    def run(self, fixture_slug: str) -> BenchmarkResult:
        paths = resolve_fixture(fixture_slug)
        if not paths.photo_exists:
            raise FileNotFoundError(f"fixture photo missing: {paths.photo}")
        image_bytes = paths.photo.read_bytes()
        output = self._backend.segment(image_bytes)
        stages: list[BaseStageResult] = [
            BaseStageResult(
                label="upload",
                kind="upload",
                endpoint=None,
                duration_ms=round(output.upload_ms * 10.0) / 10.0,
                price_usd=0.0,
            )
        ]
        for stage in output.stages:
            stages.append(
                BaseStageResult(
                    label=stage.label,
                    kind=_salad_stage_kind(stage.label),
                    endpoint=None,
                    duration_ms=round(stage.latency_ms * 10.0) / 10.0,
                    price_usd=0.0,
                    ok=stage.success,
                    detail=stage.error,
                )
            )
        return BenchmarkResult(
            fixture=fixture_slug,
            backend="salad",
            replay=False,
            image_url="",
            stages=stages,
            fal_calls=[],
            api_cost_usd=0.0,
            total_ms=round(output.total_ms * 10.0) / 10.0,
            wall_height_cm=output.wall_height_cm,
            wall_height_source=output.wall_height_source,
        )

    def close(self) -> None:
        self._backend.close()
