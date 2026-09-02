"""Cost model: fal per-call API pricing vs Salad GPU-hour allocation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRICING_PATH = REPO_ROOT / "configs" / "pricing.yaml"


@dataclass(frozen=True)
class PricingConfig:
    fal_sam3_usd_per_call: float = 0.005
    fal_depth_usd_per_call: float = 0.0
    salad_gpu_class: str = "rtx_4090"
    salad_gpu_hour_usd: float = 0.16


def _extract_float(text: str, key: str, default: float) -> float:
    match = re.search(rf"^\s*{re.escape(key)}:\s*([0-9.]+)\s*$", text, re.MULTILINE)
    return float(match.group(1)) if match else default


def _extract_str(text: str, key: str, default: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}:\s*(\S+)\s*$", text, re.MULTILINE)
    return match.group(1) if match else default


def load_pricing(path: Path | None = None) -> PricingConfig:
    """Load pricing from configs/pricing.yaml (stdlib parser; no PyYAML required)."""
    pricing_path = path or DEFAULT_PRICING_PATH
    if not pricing_path.is_file():
        return PricingConfig()

    text = pricing_path.read_text(encoding="utf-8")
    return PricingConfig(
        fal_sam3_usd_per_call=_extract_float(text, "sam3_usd_per_call", 0.005),
        fal_depth_usd_per_call=_extract_float(text, "depth_usd_per_call", 0.0),
        salad_gpu_class=_extract_str(text, "gpu_class", "rtx_4090"),
        salad_gpu_hour_usd=_extract_float(text, "gpu_hour_usd", 0.16),
    )


def fal_api_cost_usd(
    *,
    sam3_calls: int = 4,
    depth_calls: int = 1,
    pricing: PricingConfig | None = None,
) -> float:
    """C1: fal API cost from per-call pricing (4× SAM3 + depth)."""
    cfg = pricing or load_pricing()
    return (
        sam3_calls * cfg.fal_sam3_usd_per_call + depth_calls * cfg.fal_depth_usd_per_call
    )


def salad_allocated_cost_usd(
    *,
    active_seconds: float,
    pricing: PricingConfig | None = None,
) -> float:
    """C2: Salad allocated cost from GPU-hour rate × wall time."""
    cfg = pricing or load_pricing()
    hours = max(0.0, active_seconds) / 3600.0
    return hours * cfg.salad_gpu_hour_usd


def cost_per_1k_images(cost_usd: float) -> float:
    """C3: headline cost normalized per 1k images."""
    return float(cost_usd) * 1000.0
