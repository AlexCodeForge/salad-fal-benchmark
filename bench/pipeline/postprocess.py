"""Local CPU post-process stub — VLM wall height fixed at 260 cm."""

from __future__ import annotations

from dataclasses import dataclass

from bench.fal.constants import WALL_HEIGHT_STUB_CM, WALL_HEIGHT_STUB_SOURCE


@dataclass(frozen=True)
class PostprocessOutput:
    wall_height_cm: float
    wall_height_source: str


def apply_postprocess() -> PostprocessOutput:
    """Stub postprocess matching Rust analyze scope (no VLM, no mask collapse)."""
    return PostprocessOutput(
        wall_height_cm=WALL_HEIGHT_STUB_CM,
        wall_height_source=WALL_HEIGHT_STUB_SOURCE,
    )
