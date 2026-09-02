"""Benchmark metrics: latency, cost, and quality."""

from bench.metrics.cost import (
    PricingConfig,
    cost_per_1k_images,
    fal_api_cost_usd,
    load_pricing,
    salad_allocated_cost_usd,
)
from bench.metrics.latency import percentile, summarize_latency, tag_run_temperature
from bench.metrics.quality import (
    depth_pearson_r,
    depth_spearman_r,
    evaluate_quality_gates,
    mask_iou,
    mask_iou_mean,
    mask_iou_min,
)

__all__ = [
    "PricingConfig",
    "cost_per_1k_images",
    "depth_pearson_r",
    "depth_spearman_r",
    "evaluate_quality_gates",
    "fal_api_cost_usd",
    "load_pricing",
    "mask_iou",
    "mask_iou_mean",
    "mask_iou_min",
    "percentile",
    "salad_allocated_cost_usd",
    "summarize_latency",
    "tag_run_temperature",
]
