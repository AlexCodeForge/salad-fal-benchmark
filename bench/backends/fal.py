"""fal.ai benchmark backend — sequential 4× SAM3 + depth (Rust segment_fal parity)."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from bench.backends.base import BenchmarkResult, StageResult
from bench.fal.client import FalCallTracker, FalClient, FalClientConfig
from bench.fal.constants import (
    DEPTH_ENDPOINT,
    FLOOR_SAM3_PROMPTS,
    MAX_FLOOR_MASKS,
    MAX_FLOOR_SAM3_EXTRA_MASKS,
    MAX_WALL_MASKS,
    MAX_WALL_SAM3_EXTRA_MASKS,
    SAM3_IMAGE_ENDPOINT,
    WALL_HEIGHT_STUB_CM,
    WALL_HEIGHT_STUB_SOURCE,
    WALL_SAM3_PROMPTS,
)
from bench.fal.replay import resolve_fixture_dir, resolve_photo_path, resolve_replay_dir


class FalBackend:
    def __init__(self, *, replay: bool = False) -> None:
        self.replay = replay

    def run(self, fixture_slug: str) -> BenchmarkResult:
        fixture_dir = resolve_fixture_dir(fixture_slug)
        photo_path = resolve_photo_path(fixture_dir)
        image_bytes = photo_path.read_bytes()

        replay_dir = resolve_replay_dir(fixture_dir) if self.replay else None
        client_config = FalClientConfig(replay=self.replay, replay_dir=replay_dir)
        if self.replay and not client_config.replay_dir:
            client_config.replay_dir = replay_dir
            client_config.replay = True

        tracker = FalCallTracker()
        t0 = time.perf_counter()
        stages: list[StageResult] = []

        with FalClient(client_config) as client:
            upload_start = time.perf_counter()
            image_url = client.upload_image_bytes(image_bytes, "image/jpeg")
            upload_ms = (time.perf_counter() - upload_start) * 1000.0
            stages.append(
                StageResult(
                    label="upload",
                    kind="upload",
                    endpoint=None,
                    duration_ms=round(upload_ms * 10.0) / 10.0,
                    price_usd=0.0,
                )
            )

            for i, prompt in enumerate(WALL_SAM3_PROMPTS):
                max_masks = MAX_WALL_MASKS if i == 0 else MAX_WALL_SAM3_EXTRA_MASKS
                stage = self._run_sam3(
                    client,
                    tracker,
                    image_url=image_url,
                    prompt=prompt,
                    class_name="wall",
                    max_masks=max_masks,
                )
                stages.append(stage)

            for i, prompt in enumerate(FLOOR_SAM3_PROMPTS):
                max_masks = MAX_FLOOR_MASKS if i == 0 else MAX_FLOOR_SAM3_EXTRA_MASKS
                stage = self._run_sam3(
                    client,
                    tracker,
                    image_url=image_url,
                    prompt=prompt,
                    class_name="floor",
                    max_masks=max_masks,
                )
                stages.append(stage)

            depth_stage = self._run_depth(client, tracker, image_url)
            stages.append(depth_stage)

        total_ms = round((time.perf_counter() - t0) * 1000.0 * 10.0) / 10.0
        fal_calls = [asdict(c) for c in tracker.calls]

        return BenchmarkResult(
            fixture=fixture_slug,
            backend="fal",
            replay=self.replay,
            image_url=image_url,
            stages=stages,
            fal_calls=fal_calls,
            api_cost_usd=tracker.cost_usd(),
            total_ms=total_ms,
            wall_height_cm=WALL_HEIGHT_STUB_CM,
            wall_height_source=WALL_HEIGHT_STUB_SOURCE,
        )

    def _run_sam3(
        self,
        client: FalClient,
        tracker: FalCallTracker,
        *,
        image_url: str,
        prompt: str,
        class_name: str,
        max_masks: int,
    ) -> StageResult:
        label = f"{class_name}:{prompt}:sam3"
        args: dict[str, Any] = {
            "image_url": image_url,
            "prompt": prompt,
            "apply_mask": False,
            "return_multiple_masks": True,
            "max_masks": max_masks,
            "include_scores": True,
            "include_boxes": True,
        }
        t0 = time.perf_counter()
        try:
            client.subscribe(SAM3_IMAGE_ENDPOINT, args, label, tracker)
            ok = True
            detail = None
        except Exception as exc:  # noqa: BLE001 — stage failure recorded, not fatal in replay tests
            ok = False
            detail = str(exc)
        duration_ms = round((time.perf_counter() - t0) * 1000.0 * 10.0) / 10.0
        price = next((c.price_usd for c in reversed(tracker.calls) if c.label == label), 0.0)
        return StageResult(
            label=label,
            kind="sam3",
            endpoint=SAM3_IMAGE_ENDPOINT,
            duration_ms=duration_ms,
            price_usd=price,
            ok=ok,
            detail=detail,
        )

    def _run_depth(
        self,
        client: FalClient,
        tracker: FalCallTracker,
        image_url: str,
    ) -> StageResult:
        label = "depth"
        args = {"image_url": image_url}
        t0 = time.perf_counter()
        try:
            client.subscribe(DEPTH_ENDPOINT, args, label, tracker)
            ok = True
            detail = None
        except Exception as exc:  # noqa: BLE001
            ok = False
            detail = str(exc)
        duration_ms = round((time.perf_counter() - t0) * 1000.0 * 10.0) / 10.0
        price = next((c.price_usd for c in reversed(tracker.calls) if c.label == label), 0.0)
        return StageResult(
            label=label,
            kind="depth",
            endpoint=DEPTH_ENDPOINT,
            duration_ms=duration_ms,
            price_usd=price,
            ok=ok,
            detail=detail,
        )
