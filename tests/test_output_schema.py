"""Output schema tests: JSON report shape and CSV headline columns."""

import csv
import json
from pathlib import Path

import pytest

from bench.metrics.cost import fal_api_cost_usd, load_pricing
from bench.output.csv_writer import SUMMARY_COLUMNS, write_compare_csv, write_summary_csv
from bench.output.json_writer import build_report, write_report_json


@pytest.fixture
def sample_row() -> dict:
    return {
        "fixture": "terminados-02",
        "backend": "fal",
        "run_id": 1,
        "run_temperature": "warm",
        "label": "wall:wall:sam3",
        "L1_client_e2e_ms": 1200.0,
        "L2_upload_ms": 150.0,
        "L3_sam3_total_ms": 800.0,
        "L5_depth_ms": 200.0,
        "L6_queue_wait_ms": 50.0,
        "C1_api_cost_usd": 0.005,
        "C2_allocated_cost_usd": 0.0,
        "C3_cost_per_1k_images": 5.0,
        "Q1_mask_iou_mean": 0.99,
        "Q2_mask_iou_min": 0.985,
        "Q3_depth_pearson_r": 0.97,
        "Q4_depth_spearman_r": 0.96,
    }


def test_pricing_yaml_values(repo_root: Path):
    pricing = load_pricing(repo_root / "configs" / "pricing.yaml")
    assert pricing.fal_sam3_usd_per_call == pytest.approx(0.005)
    assert pricing.salad_gpu_hour_usd == pytest.approx(0.16)
    assert pricing.salad_gpu_class == "rtx_4090"
    assert fal_api_cost_usd(sam3_calls=4, depth_calls=1, pricing=pricing) == pytest.approx(
        0.02
    )


def test_json_report_schema(sample_row: dict, tmp_path: Path):
    report = build_report(
        fixture="terminados-02",
        backend="fal",
        rows=[sample_row],
        summary={"L1_client_e2e_ms": {"p50": 1200.0, "p95": 1300.0, "n": 1}},
    )
    out = tmp_path / "report.json"
    write_report_json(out, report)

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["version"] == "0.1.0"
    assert loaded["fixture"] == "terminados-02"
    assert loaded["backend"] == "fal"
    assert isinstance(loaded["rows"], list)
    assert loaded["rows"][0]["label"] == "wall:wall:sam3"
    assert "generated_at" in loaded


def test_summary_csv_columns(sample_row: dict, tmp_path: Path):
    out = tmp_path / "summary.csv"
    write_summary_csv(out, [sample_row])

    with out.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(SUMMARY_COLUMNS)
        row = next(reader)
        assert row["C1_api_cost_usd"] == "0.005"
        assert row["Q2_mask_iou_min"] == "0.985"
        assert row["L1_client_e2e_ms"] == "1200.0"


def test_compare_csv_delta(sample_row: dict, tmp_path: Path):
    fal_row = {**sample_row, "backend": "fal", "L1_client_e2e_ms": 1000.0}
    salad_row = {
        **sample_row,
        "backend": "salad",
        "L1_client_e2e_ms": 900.0,
        "C1_api_cost_usd": 0.0,
        "C2_allocated_cost_usd": 0.004,
    }
    out = tmp_path / "compare.csv"
    write_compare_csv(out, fal_rows=[fal_row], salad_rows=[salad_row])

    with out.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)
        assert row["delta_L1_ms"] == "-100.0"
        assert row["fal_C1_api_cost_usd"] == "0.005"
        assert row["salad_C2_allocated_cost_usd"] == "0.004"
