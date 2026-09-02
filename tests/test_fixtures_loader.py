"""Tests for bench.fixtures.loader."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from bench.cli import app
from bench.fixtures.loader import (
    fixtures_root,
    list_fixtures,
    load_fal_replay_manifest,
    resolve_fixture,
)

runner = CliRunner()


def test_tier_a_has_terminados_02():
    tier_a = list_fixtures(tier="A")
    assert len(tier_a) == 1
    assert tier_a[0].id == "terminados-02"
    assert tier_a[0].replay is True


def test_tier_b_has_ten_scenes():
    tier_b = list_fixtures(tier="B")
    assert len(tier_b) == 10
    ids = {entry.id for entry in tier_b}
    assert "terminados-02" in ids
    assert "terminados-03" in ids
    assert "terminados-28" in ids


def test_terminados_02_photo_exists(repo_root: Path):
    paths = resolve_fixture("terminados-02")
    assert paths.photo == fixtures_root() / "terminados-02" / "photo.jpg"
    assert paths.photo.is_file()
    assert paths.photo.stat().st_size > 0


def test_terminados_02_fal_replay_manifest(repo_root: Path):
    manifest = load_fal_replay_manifest("terminados-02")
    labels = [call["label"] for call in manifest["calls"]]
    assert labels == [
        "wall:wall:sam3",
        "wall:molding:sam3",
        "wall:mullion:sam3",
        "floor:floor:sam3",
        "depth",
        "room-height:gemini-2.5-flash",
    ]
    paths = resolve_fixture("terminados-02")
    assert paths.replay_ready is True
    assert (paths.fal_replay_dir / "depth.png").is_file()
    assert (paths.fal_replay_dir / "masks" / "wall_1.png").is_file()


def test_resolve_unknown_fixture_raises():
    with pytest.raises(KeyError, match="unknown-fixture"):
        resolve_fixture("unknown-fixture")


def test_cli_list_fixtures_shows_tier_a_and_b():
    result = runner.invoke(app, ["list-fixtures"])
    assert result.exit_code == 0
    assert "Tier A:" in result.stdout
    assert "Tier B:" in result.stdout
    assert "terminados-02" in result.stdout
    assert "photo=ok" in result.stdout
    assert "replay=ok" in result.stdout


def test_cli_list_fixtures_tier_filter():
    result = runner.invoke(app, ["list-fixtures", "--tier", "A"])
    assert result.exit_code == 0
    assert "Tier A:" in result.stdout
    assert "Tier B:" not in result.stdout
