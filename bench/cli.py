"""Typer CLI for salad-fal-benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from bench.config import get_settings
from bench.fixtures.loader import format_fixture_line, list_fixtures, resolve_fixture
from bench.runner import default_output_dir, run_benchmark

app = typer.Typer(
    name="bench",
    help="Benchmark fal.ai vs Salad GPU for Rust-prod analyze preprocess.",
)

_MILESTONE_VARS: dict[str, tuple[tuple[str, bool], ...]] = {
    "m0": (),
    "m1": (("FAL_KEY", False),),
    "m1_replay": (),
    "m4": (
        ("SALAD_API_KEY", True),
        ("SALAD_ORGANIZATION_NAME", False),
        ("SALAD_PROJECT_NAME", False),
    ),
    "m5": (
        ("FAL_KEY", False),
        ("SALAD_API_KEY", True),
        ("SALAD_ANALYZE_GATEWAY_URL", False),
    ),
}


@app.command("list-fixtures")
def list_fixtures_cmd(
    tier: Optional[str] = typer.Option(
        None,
        "--tier",
        help="Filter by tier: A, B, or C",
    ),
) -> None:
    """List available benchmark fixtures (Tier A/B/C)."""
    tier_filter = tier.upper() if tier else None
    if tier_filter is not None and tier_filter not in {"A", "B", "C"}:
        raise typer.BadParameter("--tier must be A, B, or C")
    entries = list_fixtures(tier=tier_filter)  # type: ignore[arg-type]
    if not entries:
        typer.echo("No fixtures found.")
        return
    current_tier: str | None = None
    for entry in entries:
        if entry.tier != current_tier:
            current_tier = entry.tier
            typer.echo(f"Tier {current_tier}:")
        paths = resolve_fixture(entry.id, tier=entry.tier)
        typer.echo(f"  {format_fixture_line(paths)}")


@app.command()
def run(
    backend: str = typer.Option("fal", help="Backend: fal, salad, or both"),
    fixture: str = typer.Option(..., help="Fixture slug, e.g. terminados-02"),
    replay: bool = typer.Option(False, help="Use fal replay fixtures (no FAL_KEY)"),
    runs: int = typer.Option(1, help="Number of benchmark runs"),
    stage_mode: str = typer.Option("sequential", help="Stage mode: sequential or parallel"),
    output: str | None = typer.Option(None, "--output", "-o", help="Write JSON result to path"),
) -> None:
    """Run benchmark for a fixture against the selected backend."""
    backend_norm = backend.lower().strip()
    if backend_norm not in {"fal", "salad", "both"}:
        raise typer.BadParameter("--backend must be fal, salad, or both")

    stage_mode_norm = stage_mode.lower().strip()
    if stage_mode_norm not in {"sequential", "parallel"}:
        raise typer.BadParameter("--stage-mode must be sequential or parallel")

    if runs < 1:
        raise typer.BadParameter("--runs must be >= 1")

    try:
        artifacts = run_benchmark(
            backend_norm,  # type: ignore[arg-type]
            fixture,
            replay=replay,
            runs=runs,
            stage_mode=stage_mode_norm,  # type: ignore[arg-type]
        )
    except (ValueError, NotImplementedError, FileNotFoundError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    primary_backend = "fal" if backend_norm in {"fal", "both"} else "salad"
    report = artifacts.reports[primary_backend]

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if runs == 1 and backend_norm != "both":
            legacy = artifacts.primary_result(primary_backend)
            payload = legacy.to_dict() if legacy else report
        else:
            payload = report
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        typer.echo(f"Wrote {out_path}")
    else:
        typer.echo(json.dumps(report, indent=2))

    typer.echo(f"Artifacts written to {artifacts.output_dir}", err=True)


@app.command()
def report(
    input_path: str = typer.Option(..., "--input", help="Benchmark JSON output to summarize"),
) -> None:
    """Generate summary report from benchmark JSON artifacts."""
    path = Path(input_path)
    if not path.is_file():
        typer.echo(f"Input not found: {path}", err=True)
        raise typer.Exit(code=1)

    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    latency = summary.get("latency", {})
    cost = summary.get("cost", {})
    typer.echo(f"fixture={payload.get('fixture')} backend={payload.get('backend')}")
    typer.echo(
        f"L1 p50={latency.get('p50', 'n/a')} p95={latency.get('p95', 'n/a')} n={latency.get('n', 0)}"
    )
    typer.echo(
        f"C1={cost.get('C1_api_cost_usd', 'n/a')} "
        f"C2={cost.get('C2_allocated_cost_usd', 'n/a')} "
        f"C3={cost.get('C3_cost_per_1k_images', 'n/a')}"
    )


@app.command("validate-env")
def validate_env(
    milestone: str = typer.Option("m0", help="Milestone gate: m0, m1, m4, m5"),
    replay: bool = typer.Option(False, help="Use m1_replay gate (no secrets)"),
) -> None:
    """Validate required environment variables for a milestone."""
    import os

    gate = milestone.lower().strip()
    if replay and gate == "m1":
        gate = "m1_replay"
    if gate not in _MILESTONE_VARS:
        raise typer.BadParameter(f"unknown milestone {milestone!r}; use m0, m1, m4, m5")

    settings = get_settings()
    env_map = {
        "FAL_KEY": settings.fal_key or os.environ.get("FAL_KEY", ""),
        "SALAD_API_KEY": settings.salad_api_key or os.environ.get("SALAD_API_KEY", ""),
        "SALAD_ORGANIZATION_NAME": settings.salad_organization_name
        or os.environ.get("SALAD_ORGANIZATION_NAME", ""),
        "SALAD_PROJECT_NAME": settings.salad_project_name
        or os.environ.get("SALAD_PROJECT_NAME", ""),
        "SALAD_ANALYZE_GATEWAY_URL": settings.resolved_analyze_gateway_url()
        or os.environ.get("SALAD_ANALYZE_GATEWAY_URL", "")
        or os.environ.get("SALAD_GATEWAY_URL", ""),
        "SALAD_SAM3_GATEWAY_URL": settings.salad_sam3_gateway_url
        or os.environ.get("SALAD_SAM3_GATEWAY_URL", ""),
        "SALAD_DEPTH_GATEWAY_URL": settings.salad_depth_gateway_url
        or os.environ.get("SALAD_DEPTH_GATEWAY_URL", ""),
    }

    missing: list[str] = []
    for var_name, secret in _MILESTONE_VARS[gate]:
        value = env_map.get(var_name, os.environ.get(var_name, ""))
        if not str(value).strip():
            missing.append(f"{var_name}{' (secret)' if secret else ''}")

    if missing:
        typer.echo(f"milestone {gate}: missing {', '.join(missing)}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"milestone {gate}: ok (output_dir={default_output_dir()})")
