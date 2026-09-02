"""CSV writers for flat stage rows and fal vs salad comparison."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Sequence

SUMMARY_COLUMNS: tuple[str, ...] = (
    "fixture",
    "backend",
    "run_id",
    "run_temperature",
    "label",
    "L1_client_e2e_ms",
    "L2_upload_ms",
    "L3_sam3_total_ms",
    "L5_depth_ms",
    "L6_queue_wait_ms",
    "C1_api_cost_usd",
    "C2_allocated_cost_usd",
    "C3_cost_per_1k_images",
    "Q1_mask_iou_mean",
    "Q2_mask_iou_min",
    "Q3_depth_pearson_r",
    "Q4_depth_spearman_r",
)

COMPARE_COLUMNS: tuple[str, ...] = (
    "fixture",
    "label",
    "fal_L1_client_e2e_ms",
    "salad_L1_client_e2e_ms",
    "delta_L1_ms",
    "fal_C1_api_cost_usd",
    "salad_C2_allocated_cost_usd",
    "Q2_mask_iou_min",
    "Q3_depth_pearson_r",
)


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_summary_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write flat stage rows with L/C/Q headline columns."""
    _write_csv(path, SUMMARY_COLUMNS, rows)


def write_compare_csv(
    path: Path,
    *,
    fal_rows: Iterable[dict[str, Any]],
    salad_rows: Iterable[dict[str, Any]],
) -> None:
    """Write fal vs salad delta rows keyed by fixture + label."""
    fal_by_label = {(r.get("fixture"), r.get("label")): r for r in fal_rows}
    compare_rows: list[dict[str, Any]] = []

    for salad_row in salad_rows:
        key = (salad_row.get("fixture"), salad_row.get("label"))
        fal_row = fal_by_label.get(key, {})
        fal_l1 = fal_row.get("L1_client_e2e_ms")
        salad_l1 = salad_row.get("L1_client_e2e_ms")
        delta_l1 = ""
        if fal_l1 not in (None, "") and salad_l1 not in (None, ""):
            delta_l1 = float(salad_l1) - float(fal_l1)

        compare_rows.append(
            {
                "fixture": salad_row.get("fixture", ""),
                "label": salad_row.get("label", ""),
                "fal_L1_client_e2e_ms": fal_l1 if fal_l1 is not None else "",
                "salad_L1_client_e2e_ms": salad_l1 if salad_l1 is not None else "",
                "delta_L1_ms": delta_l1,
                "fal_C1_api_cost_usd": fal_row.get("C1_api_cost_usd", ""),
                "salad_C2_allocated_cost_usd": salad_row.get("C2_allocated_cost_usd", ""),
                "Q2_mask_iou_min": salad_row.get("Q2_mask_iou_min", ""),
                "Q3_depth_pearson_r": salad_row.get("Q3_depth_pearson_r", ""),
            }
        )

    _write_csv(path, COMPARE_COLUMNS, compare_rows)
