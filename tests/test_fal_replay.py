"""Tier A fal replay — no FAL_KEY required."""

from __future__ import annotations

import json
import shutil
import sys
from subprocess import run as subprocess_run

import pytest

from bench.backends.registry import get_backend
from bench.fal.replay import load_manifest, resolve_replay_dir, should_skip_replay_label


def _bench_argv(*args: str) -> list[str]:
    bench_bin = shutil.which("bench")
    if bench_bin:
        return [bench_bin, *args]
    return [sys.executable, "-m", "bench", *args]


EXPECTED_INFERENCE_LABELS = (
    "wall:wall:sam3",
    "wall:molding:sam3",
    "wall:mullion:sam3",
    "floor:floor:sam3",
    "depth",
)


def test_replay_manifest_skips_vlm(repo_root):
    fixture_dir = repo_root / "fixtures" / "terminados-02"
    if not fixture_dir.is_dir():
        pytest.skip("terminados-02 fixtures not present (pkg-fixtures)")
    replay_dir = resolve_replay_dir(fixture_dir)
    manifest = load_manifest(replay_dir)
    labels = [c.label for c in manifest.calls]
    assert "room-height:gemini-2.5-flash" in labels
    indexed = manifest.index(replay_dir)
    assert "room-height:gemini-2.5-flash" not in indexed
    assert all(
        should_skip_replay_label(c.label, c.endpoint)
        for c in manifest.calls
        if c.label.startswith("room-height:")
    )


def test_fal_replay_run_stage_labels(repo_root):
    fixture_dir = repo_root / "fixtures" / "terminados-02"
    if not fixture_dir.is_dir():
        pytest.skip("terminados-02 fixtures not present (pkg-fixtures)")

    backend = get_backend("fal", replay=True)
    result = backend.run("terminados-02")

    assert result.backend == "fal"
    assert result.replay is True
    assert result.wall_height_cm == 260.0
    assert result.wall_height_source == "stub"

    stage_labels = result.stage_labels()
    assert stage_labels[0] == "upload"
    for label in EXPECTED_INFERENCE_LABELS:
        assert label in stage_labels

    fal_call_labels = [c["label"] for c in result.fal_calls]
    assert fal_call_labels == list(EXPECTED_INFERENCE_LABELS)
    assert "room-height:gemini-2.5-flash" not in fal_call_labels

    assert result.api_cost_usd == pytest.approx(0.02, abs=0.001)
    assert all(s.ok for s in result.stages)


def test_fal_replay_cli(repo_root, tmp_path):
    fixture_dir = repo_root / "fixtures" / "terminados-02"
    if not fixture_dir.is_dir():
        pytest.skip("terminados-02 fixtures not present (pkg-fixtures)")

    out_file = tmp_path / "result.json"
    proc = subprocess_run(
        _bench_argv(
            "run",
            "--backend",
            "fal",
            "--fixture",
            "terminados-02",
            "--replay",
            "--output",
            str(out_file),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout

    payload = json.loads(out_file.read_text(encoding="utf-8"))
    labels = [s["label"] for s in payload["stages"]]
    assert labels[0] == "upload"
    for label in EXPECTED_INFERENCE_LABELS:
        assert label in labels
    assert payload["api_cost_usd"] == pytest.approx(0.02, abs=0.001)
