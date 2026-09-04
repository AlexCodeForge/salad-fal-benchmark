"""Optional lab scope: wall + floor SAM3 only (no molding/mullion)."""

from __future__ import annotations

import os

from bench.fal.constants import (
    FLOOR_SAM3_PROMPTS,
    MAX_FLOOR_MASKS,
    MAX_FLOOR_SAM3_EXTRA_MASKS,
    MAX_WALL_MASKS,
    MAX_WALL_SAM3_EXTRA_MASKS,
    WALL_SAM3_PROMPTS,
)

WALL_SAM3_STAGES_FULL: tuple[tuple[str, str, str, int], ...] = (
    ("wall:wall:sam3", "wall", "wall", 8),
    ("wall:molding:sam3", "wall", "molding", 4),
    ("wall:mullion:sam3", "wall", "mullion", 4),
)
FLOOR_SAM3_STAGE = ("floor:floor:sam3", "floor", "floor", 8)


def lab_wall_floor_only() -> bool:
    return os.environ.get("BENCH_LAB_WALL_FLOOR_ONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def wall_sam3_prompts() -> tuple[str, ...]:
    if lab_wall_floor_only():
        return ("wall",)
    return WALL_SAM3_PROMPTS


def wall_sam3_stages() -> tuple[tuple[str, str, str, int], ...]:
    if lab_wall_floor_only():
        return (("wall:wall:sam3", "wall", "wall", 8),)
    return WALL_SAM3_STAGES_FULL


def floor_sam3_prompts() -> tuple[str, ...]:
    return FLOOR_SAM3_PROMPTS


def max_wall_masks(index: int) -> int:
    return MAX_WALL_MASKS if index == 0 else MAX_WALL_SAM3_EXTRA_MASKS


def max_floor_masks(index: int) -> int:
    return MAX_FLOOR_MASKS if index == 0 else MAX_FLOOR_SAM3_EXTRA_MASKS
