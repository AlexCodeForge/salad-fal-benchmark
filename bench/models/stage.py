"""Stage graph models loaded from configs/stages.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGES_PATH = REPO_ROOT / "configs" / "stages.yaml"


@dataclass(frozen=True)
class StageDef:
    id: str
    label: str
    kind: str
    endpoint: str | None = None
    class_name: str | None = None
    prompt: str | None = None
    max_masks: int | None = None
    min_score: float | None = None
    depends_on: tuple[str, ...] = ()
    wall_height_cm: float | None = None
    wall_height_source: str | None = None


@dataclass(frozen=True)
class StageGraph:
    mode: str
    stages: tuple[StageDef, ...]

    def inference_stages(self) -> tuple[StageDef, ...]:
        return tuple(s for s in self.stages if s.kind in {"sam3", "depth"})

    def labels(self) -> list[str]:
        return [s.label for s in self.stages]


def _parse_stage(raw: dict[str, Any]) -> StageDef:
    depends = raw.get("depends_on") or []
    return StageDef(
        id=str(raw["id"]),
        label=str(raw["label"]),
        kind=str(raw["kind"]),
        endpoint=raw.get("endpoint"),
        class_name=raw.get("class"),
        prompt=raw.get("prompt"),
        max_masks=int(raw["max_masks"]) if "max_masks" in raw else None,
        min_score=float(raw["min_score"]) if "min_score" in raw else None,
        depends_on=tuple(str(d) for d in depends),
        wall_height_cm=float(raw["wall_height_cm"]) if "wall_height_cm" in raw else None,
        wall_height_source=raw.get("wall_height_source"),
    )


def load_stage_graph(path: Path | None = None) -> StageGraph:
    """Load Rust-prod sequential stage DAG from configs/stages.yaml."""
    stages_path = path or DEFAULT_STAGES_PATH
    with stages_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid stages config: {stages_path}")

    raw_stages = data.get("stages", [])
    if not isinstance(raw_stages, list):
        raise ValueError(f"stages must be a list in {stages_path}")

    return StageGraph(
        mode=str(data.get("mode", "sequential")),
        stages=tuple(_parse_stage(row) for row in raw_stages if isinstance(row, dict)),
    )
