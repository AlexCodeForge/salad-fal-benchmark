"""Benchmark runner — N runs, latency aggregation, artifact writers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from bench.backends.base import BenchmarkResult
from bench.config import get_settings
from bench.metrics.cost import cost_per_1k_images, fal_api_cost_usd, load_pricing, salad_allocated_cost_usd
from bench.metrics.latency import summarize_latency, tag_run_temperature
from bench.models.report import RunRecord, RunSummary
from bench.output.csv_writer import write_compare_csv, write_summary_csv
from bench.output.json_writer import build_report, write_report_json
from bench.pipeline.executor import PipelineExecutor, StageMode

BackendName = Literal["fal", "salad", "both"]


@dataclass
class BenchmarkArtifacts:
    fixture: str
    backends: list[str]
    runs: int
    records: list[RunRecord] = field(default_factory=list)
    reports: dict[str, dict[str, Any]] = field(default_factory=dict)
    output_dir: Path = field(default_factory=Path)

    def primary_result(self, backend: str) -> BenchmarkResult | None:
        for record in self.records:
            if record.result.backend == backend:
                return record.result
        return None


def default_output_dir() -> Path:
    settings = get_settings()
    if settings.bench_output_dir:
        return Path(settings.bench_output_dir)
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "out"


def _stage_kind_totals(stages: list) -> dict[str, float]:
    sam3_ms = sum(s.duration_ms for s in stages if s.kind == "sam3")
    depth_ms = next((s.duration_ms for s in stages if s.kind == "depth"), 0.0)
    upload_ms = next((s.duration_ms for s in stages if s.kind == "upload"), 0.0)
    return {"sam3_total_ms": sam3_ms, "depth_ms": depth_ms, "upload_ms": upload_ms}


def _queue_wait_ms(result: BenchmarkResult) -> float:
    total = 0.0
    for call in result.fal_calls:
        queue = call.get("queue_wait_ms")
        if queue is not None:
            total += float(queue)
    return total


def result_to_rows(
    result: BenchmarkResult,
    *,
    run_id: int,
    run_temperature: str,
    pricing=None,
) -> list[dict[str, Any]]:
    """Flatten a BenchmarkResult into CSV headline rows (one per stage)."""
    pricing = pricing or load_pricing()
    totals = _stage_kind_totals(result.stages)
    queue_ms = _queue_wait_ms(result)

    if result.backend == "fal":
        c1 = result.api_cost_usd
        c2 = 0.0
    else:
        c1 = 0.0
        active_s = result.total_ms / 1000.0
        c2 = salad_allocated_cost_usd(active_seconds=active_s, pricing=pricing)

    headline_cost = c1 if result.backend == "fal" else c2
    c3 = cost_per_1k_images(headline_cost)

    rows: list[dict[str, Any]] = []
    for stage in result.stages:
        rows.append(
            {
                "fixture": result.fixture,
                "backend": result.backend,
                "run_id": run_id,
                "run_temperature": run_temperature,
                "label": stage.label,
                "L1_client_e2e_ms": result.total_ms,
                "L2_upload_ms": totals["upload_ms"],
                "L3_sam3_total_ms": totals["sam3_total_ms"],
                "L5_depth_ms": totals["depth_ms"],
                "L6_queue_wait_ms": queue_ms if result.backend == "fal" else "",
                "C1_api_cost_usd": c1,
                "C2_allocated_cost_usd": c2,
                "C3_cost_per_1k_images": c3,
                "Q1_mask_iou_mean": "",
                "Q2_mask_iou_min": "",
                "Q3_depth_pearson_r": "",
                "Q4_depth_spearman_r": "",
                "duration_ms": stage.duration_ms,
                "stage_kind": stage.kind,
                "ok": stage.ok,
            }
        )
    return rows


def _aggregate_summary(
    fixture: str,
    backend: str,
    records: list[RunRecord],
) -> RunSummary:
    e2e_values = [r.result.total_ms for r in records]
    latency = summarize_latency(
        [{"client_e2e_ms": v} for v in e2e_values],
        field="client_e2e_ms",
    )
    pricing = load_pricing()
    if backend == "fal":
        cost_val = fal_api_cost_usd(pricing=pricing)
        cost = {
            "C1_api_cost_usd": cost_val,
            "C2_allocated_cost_usd": 0.0,
            "C3_cost_per_1k_images": cost_per_1k_images(cost_val),
        }
    else:
        active_s = sum(r.result.total_ms for r in records) / 1000.0 / max(len(records), 1)
        c2 = salad_allocated_cost_usd(active_seconds=active_s, pricing=pricing)
        cost = {
            "C1_api_cost_usd": 0.0,
            "C2_allocated_cost_usd": c2,
            "C3_cost_per_1k_images": cost_per_1k_images(c2),
        }
    return RunSummary(
        fixture=fixture,
        backend=backend,
        runs=len(records),
        latency=latency,
        cost=cost,
    )


def run_benchmark(
    backend: BackendName,
    fixture: str,
    *,
    replay: bool = False,
    runs: int = 1,
    stage_mode: StageMode = "sequential",
    output_dir: Path | None = None,
) -> BenchmarkArtifacts:
    """Execute N benchmark runs and write JSON/CSV artifacts to out/."""
    if runs < 1:
        raise ValueError("runs must be >= 1")
    if backend == "salad" and replay:
        raise ValueError("replay mode is only supported for fal backend")
    out_dir = output_dir or default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    backend_names = ["fal", "salad"] if backend == "both" else [backend]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifacts = BenchmarkArtifacts(
        fixture=fixture,
        backends=backend_names,
        runs=runs,
        output_dir=out_dir,
    )

    all_rows_by_backend: dict[str, list[dict[str, Any]]] = {name: [] for name in backend_names}

    for backend_name in backend_names:
        executor = PipelineExecutor(
            backend_name,
            replay=replay and backend_name == "fal",
            stage_mode=stage_mode,
        )
        records: list[RunRecord] = []

        for run_id in range(1, runs + 1):
            result = executor.run(fixture)
            temperature = tag_run_temperature(run_id - 1, backend=backend_name)
            rows = result_to_rows(result, run_id=run_id, run_temperature=temperature)
            record = RunRecord(
                run_id=run_id,
                run_temperature=temperature,
                result=result,
                rows=rows,
            )
            records.append(record)
            artifacts.records.append(record)
            all_rows_by_backend[backend_name].extend(rows)

        summary = _aggregate_summary(fixture, backend_name, records)
        report = build_report(
            fixture=fixture,
            backend=backend_name,
            rows=all_rows_by_backend[backend_name],
            summary={
                "latency": summary.latency,
                "cost": summary.cost,
                "wall_height_cm": records[-1].result.wall_height_cm if records else 260.0,
                "wall_height_source": records[-1].result.wall_height_source if records else "stub",
            },
            meta={
                "replay": replay and backend_name == "fal",
                "runs": runs,
                "stage_mode": stage_mode,
            },
        )
        artifacts.reports[backend_name] = report

        json_path = out_dir / f"{timestamp}_{fixture}_{backend_name}.json"
        write_report_json(json_path, report)

    summary_csv = out_dir / f"{timestamp}_summary.csv"
    flat_rows = [row for rows in all_rows_by_backend.values() for row in rows]
    write_summary_csv(summary_csv, flat_rows)

    if backend == "both":
        compare_csv = out_dir / f"{timestamp}_compare.csv"
        write_compare_csv(
            compare_csv,
            fal_rows=all_rows_by_backend.get("fal", []),
            salad_rows=all_rows_by_backend.get("salad", []),
        )

    return artifacts
