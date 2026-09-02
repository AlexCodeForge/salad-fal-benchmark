"""FAL replay manifest loader — skips VLM room-height labels."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from bench.fal.constants import VLM_ROOM_HEIGHT_ENDPOINT

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURES_DIR = REPO_ROOT / "fixtures"

SKIP_REPLAY_LABEL_PREFIXES = ("room-height:",)


@dataclass(frozen=True)
class ReplayCall:
    label: str
    endpoint: str
    file: str


@dataclass(frozen=True)
class ReplayManifest:
    calls: tuple[ReplayCall, ...]

    def index(self, replay_dir: Path) -> dict[str, Path]:
        out: dict[str, Path] = {}
        for call in self.calls:
            if should_skip_replay_label(call.label, call.endpoint):
                continue
            out[call.label] = replay_dir / call.file
        return out


def should_skip_replay_label(label: str, endpoint: str) -> bool:
    if endpoint == VLM_ROOM_HEIGHT_ENDPOINT:
        return True
    return any(label.startswith(prefix) for prefix in SKIP_REPLAY_LABEL_PREFIXES)


def fixtures_dir() -> Path:
    env = os.environ.get("BENCH_FIXTURES_DIR", "").strip()
    return Path(env) if env else DEFAULT_FIXTURES_DIR


def resolve_fixture_dir(slug: str, base: Path | None = None) -> Path:
    path = (base or fixtures_dir()) / slug
    if not path.is_dir():
        raise FileNotFoundError(f"fixture directory not found: {path}")
    return path


def resolve_photo_path(fixture_dir: Path) -> Path:
    for name in ("photo.jpg", "photo.jpeg", "input.jpg", "input.jpeg"):
        candidate = fixture_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no photo in fixture dir: {fixture_dir}")


def resolve_replay_dir(fixture_dir: Path) -> Path:
    nested = fixture_dir / "fal_replay"
    if nested.is_dir() and (nested / "manifest.json").is_file():
        return nested
    if (fixture_dir / "manifest.json").is_file():
        return fixture_dir
    raise FileNotFoundError(f"no fal replay manifest under {fixture_dir}")


def load_manifest(replay_dir: Path) -> ReplayManifest:
    manifest_path = replay_dir / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    calls = tuple(
        ReplayCall(label=c["label"], endpoint=c["endpoint"], file=c["file"])
        for c in raw.get("calls", [])
    )
    return ReplayManifest(calls=calls)


def load_replay_index(replay_dir: Path) -> dict[str, Path]:
    manifest = load_manifest(replay_dir)
    return manifest.index(replay_dir)
