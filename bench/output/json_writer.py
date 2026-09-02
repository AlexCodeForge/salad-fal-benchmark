"""JSON report writer for benchmark runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORT_VERSION = "0.1.0"


def build_report(
    *,
    fixture: str,
    backend: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a benchmark JSON artifact with stage rows and headline summary."""
    payload: dict[str, Any] = {
        "version": REPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fixture": fixture,
        "backend": backend,
        "rows": rows,
        "summary": summary or {},
    }
    if meta:
        payload["meta"] = meta
    return payload


def write_report_json(path: Path, report: dict[str, Any]) -> None:
    """Write benchmark report JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
