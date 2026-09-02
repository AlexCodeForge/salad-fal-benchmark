"""Load benchmark fixtures from configs/fixtures.yaml and on-disk layout."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

TierName = Literal["A", "B", "C"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FIXTURES_ROOT = _REPO_ROOT / "fixtures"
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "fixtures.yaml"


def repo_root() -> Path:
    return _REPO_ROOT


def fixtures_config_path() -> Path:
    return _DEFAULT_CONFIG


def fixtures_root() -> Path:
    override = os.environ.get("BENCH_FIXTURES_DIR")
    if override:
        return Path(override)
    return _DEFAULT_FIXTURES_ROOT


@dataclass(frozen=True)
class FixtureEntry:
    """One fixture row from fixtures.yaml."""

    id: str
    tier: TierName
    scene_slug: str
    lab_image: str
    width: int
    height: int
    room_type: str
    replay: bool

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass(frozen=True)
class FixturePaths:
    """Resolved on-disk paths for a fixture id."""

    entry: FixtureEntry
    root: Path
    photo: Path
    fal_replay_dir: Path | None
    manifest_path: Path | None

    @property
    def photo_exists(self) -> bool:
        return self.photo.is_file()

    @property
    def replay_ready(self) -> bool:
        return (
            self.entry.replay
            and self.fal_replay_dir is not None
            and self.manifest_path is not None
            and self.manifest_path.is_file()
        )


def load_fixtures_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or fixtures_config_path()
    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid fixtures config (expected mapping): {config_path}")
    return data


def _tier_entries(config: dict[str, Any], tier_key: str, tier_label: TierName) -> list[FixtureEntry]:
    raw = config.get(tier_key, [])
    if not isinstance(raw, list):
        raise ValueError(f"fixtures.yaml {tier_key} must be a list")
    entries: list[FixtureEntry] = []
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError(f"Invalid fixture row in {tier_key}: {row!r}")
        entries.append(
            FixtureEntry(
                id=str(row["id"]),
                tier=tier_label,
                scene_slug=str(row["scene_slug"]),
                lab_image=str(row["lab_image"]),
                width=int(row["width"]),
                height=int(row["height"]),
                room_type=str(row["room_type"]),
                replay=bool(row.get("replay", False)),
            )
        )
    return entries


def list_fixtures(*, tier: TierName | None = None) -> list[FixtureEntry]:
    """Return fixture catalog entries, optionally filtered by tier."""
    config = load_fixtures_config()
    catalog: list[FixtureEntry] = []
    for tier_key, tier_label in (("tier_a", "A"), ("tier_b", "B"), ("tier_c", "C")):
        if tier_key not in config:
            continue
        catalog.extend(_tier_entries(config, tier_key, tier_label))
    if tier is None:
        return catalog
    return [entry for entry in catalog if entry.tier == tier]


def resolve_fixture(fixture_id: str, *, tier: TierName | None = None) -> FixturePaths:
    """Resolve fixture id to on-disk paths; raises KeyError if unknown."""
    matches = [entry for entry in list_fixtures(tier=tier) if entry.id == fixture_id]
    if not matches:
        raise KeyError(f"Unknown fixture id: {fixture_id!r}")
    # Prefer Tier A metadata when the same id appears in multiple tiers.
    entry = sorted(matches, key=lambda item: item.tier)[0]
    root = fixtures_root() / entry.id
    photo = root / "photo.jpg"
    fal_replay_dir = root / "fal_replay" if entry.replay else None
    manifest_path = fal_replay_dir / "manifest.json" if fal_replay_dir else None
    return FixturePaths(
        entry=entry,
        root=root,
        photo=photo,
        fal_replay_dir=fal_replay_dir,
        manifest_path=manifest_path,
    )


def load_fal_replay_manifest(fixture_id: str) -> dict[str, Any]:
    """Load and parse fal_replay/manifest.json for a fixture."""
    paths = resolve_fixture(fixture_id)
    if paths.manifest_path is None or not paths.manifest_path.is_file():
        raise FileNotFoundError(
            f"No fal_replay manifest for fixture {fixture_id!r} at {paths.root}"
        )
    with paths.manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or "calls" not in manifest:
        raise ValueError(f"Invalid fal_replay manifest: {paths.manifest_path}")
    return manifest


def format_fixture_line(paths: FixturePaths) -> str:
    """Human-readable one-line summary for CLI."""
    photo_flag = "photo=ok" if paths.photo_exists else "photo=missing"
    replay_flag = "replay=ok" if paths.replay_ready else "replay=no"
    return (
        f"{paths.entry.id} ({paths.entry.resolution}, {paths.entry.room_type}, "
        f"{replay_flag}, {photo_flag})"
    )
