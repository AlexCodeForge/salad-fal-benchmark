"""Pipeline executor tests — sequential prod path via fal replay."""

from __future__ import annotations

import pytest

from bench.backends.registry import get_backend
from bench.pipeline.executor import PipelineExecutor
from bench.pipeline.graph import load_pipeline_graph
from bench.pipeline.postprocess import apply_postprocess
from bench.runner import result_to_rows, run_benchmark

EXPECTED_INFERENCE_LABELS = (
    "wall:wall:sam3",
    "wall:molding:sam3",
    "wall:mullion:sam3",
    "floor:floor:sam3",
    "depth",
)


def test_stage_graph_loads_prod_dag(repo_root):
    graph = load_pipeline_graph()
    assert graph.mode == "sequential"
    labels = graph.labels()
    assert labels[0] == "upload"
    for label in EXPECTED_INFERENCE_LABELS:
        assert label in labels
    assert labels[-1] == "postprocess"


def test_postprocess_wall_height_stub():
    post = apply_postprocess()
    assert post.wall_height_cm == 260.0
    assert post.wall_height_source == "stub"


def test_executor_fal_replay(repo_root):
    fixture_dir = repo_root / "fixtures" / "terminados-02"
    if not fixture_dir.is_dir():
        pytest.skip("terminados-02 fixtures not present (pkg-fixtures)")

    executor = PipelineExecutor("fal", replay=True, stage_mode="sequential")
    result = executor.run("terminados-02")

    assert result.backend == "fal"
    assert result.replay is True
    assert result.wall_height_cm == 260.0
    assert result.wall_height_source == "stub"

    stage_labels = result.stage_labels()
    assert stage_labels[0] == "upload"
    for label in EXPECTED_INFERENCE_LABELS:
        assert label in stage_labels
    assert all(s.ok for s in result.stages)


def test_result_to_rows_headline_columns(repo_root):
    fixture_dir = repo_root / "fixtures" / "terminados-02"
    if not fixture_dir.is_dir():
        pytest.skip("terminados-02 fixtures not present (pkg-fixtures)")

    result = get_backend("fal", replay=True).run("terminados-02")
    rows = result_to_rows(result, run_id=1, run_temperature="cold")
    assert len(rows) == len(result.stages)
    row = rows[0]
    assert row["fixture"] == "terminados-02"
    assert row["backend"] == "fal"
    assert row["L1_client_e2e_ms"] == result.total_ms
    assert row["C1_api_cost_usd"] == pytest.approx(0.02, abs=0.001)


def test_runner_single_replay_run(repo_root, tmp_path):
    fixture_dir = repo_root / "fixtures" / "terminados-02"
    if not fixture_dir.is_dir():
        pytest.skip("terminados-02 fixtures not present (pkg-fixtures)")

    artifacts = run_benchmark(
        "fal",
        "terminados-02",
        replay=True,
        runs=1,
        stage_mode="sequential",
        output_dir=tmp_path,
    )
    assert artifacts.runs == 1
    assert "fal" in artifacts.reports
    report = artifacts.reports["fal"]
    assert report["summary"]["latency"]["n"] == 1
    assert len(report["rows"]) >= 6
    json_files = list(tmp_path.glob("*_terminados-02_fal.json"))
    assert len(json_files) == 1
    assert (tmp_path / json_files[0].name).is_file()
