"""Quality metrics: mask IoU and depth correlation vs fal baseline."""

from __future__ import annotations

from typing import Sequence

import numpy as np

Q2_GATE_MIN = 0.98
Q3_GATE_MEDIAN_MIN = 0.95


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union for binary masks (Q1/Q2 building block)."""
    a_bin = np.asarray(a, dtype=bool)
    b_bin = np.asarray(b, dtype=bool)
    if a_bin.shape != b_bin.shape:
        raise ValueError(f"mask shape mismatch: {a_bin.shape} vs {b_bin.shape}")

    intersection = int(np.logical_and(a_bin, b_bin).sum())
    union = int(np.logical_or(a_bin, b_bin).sum())
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def mask_iou_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def mask_iou_min(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(min(float(v) for v in values))


def depth_pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation on flattened depth maps (Q3)."""
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"depth shape mismatch: {x.shape} vs {y.shape}")
    if x.size < 2:
        return 1.0 if x.size == 1 and np.isclose(x[0], y[0]) else 0.0

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = float(np.sqrt((x_centered**2).sum() * (y_centered**2).sum()))
    if denom == 0.0:
        return 1.0 if np.allclose(x, y) else 0.0
    return float((x_centered * y_centered).sum() / denom)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def depth_spearman_r(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation on flattened depth maps (Q4)."""
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    return depth_pearson_r(_rankdata(x), _rankdata(y))


def evaluate_quality_gates(
    q2_mask_iou_min: float,
    q3_depth_pearson_values: Sequence[float],
    *,
    q2_min: float = Q2_GATE_MIN,
    q3_median_min: float = Q3_GATE_MEDIAN_MIN,
) -> dict[str, bool | float]:
    """Evaluate v1 quality gates: Q2 per fixture, Q3 corpus median."""
    q3_values = [float(v) for v in q3_depth_pearson_values]
    q3_median = percentile_median(q3_values)
    return {
        "q2_mask_iou_min": q2_mask_iou_min,
        "q2_pass": q2_mask_iou_min >= q2_min,
        "q3_depth_pearson_median": q3_median,
        "q3_pass": q3_median >= q3_median_min if q3_values else False,
        "pass": q2_mask_iou_min >= q2_min
        and (q3_median >= q3_median_min if q3_values else False),
    }


def percentile_median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
