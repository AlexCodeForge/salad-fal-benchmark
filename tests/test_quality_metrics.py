"""Unit tests for quality metrics (IoU, depth correlation, gates)."""

import numpy as np
import pytest

from bench.metrics.quality import (
    Q2_GATE_MIN,
    Q3_GATE_MEDIAN_MIN,
    depth_pearson_r,
    depth_spearman_r,
    evaluate_quality_gates,
    mask_iou,
    mask_iou_mean,
    mask_iou_min,
)


def test_mask_iou_identical():
    mask = np.array([[1, 1], [0, 0]], dtype=bool)
    assert mask_iou(mask, mask) == pytest.approx(1.0)


def test_mask_iou_disjoint():
    a = np.array([[1, 0], [0, 0]], dtype=bool)
    b = np.array([[0, 0], [0, 1]], dtype=bool)
    assert mask_iou(a, b) == pytest.approx(0.0)


def test_mask_iou_partial_overlap():
    a = np.array([[1, 1], [0, 0]], dtype=bool)
    b = np.array([[0, 1], [1, 0]], dtype=bool)
    assert mask_iou(a, b) == pytest.approx(1 / 3)


def test_mask_iou_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        mask_iou(np.zeros((2, 2), dtype=bool), np.zeros((3, 3), dtype=bool))


def test_mask_iou_aggregate_helpers():
    values = [0.99, 0.985, 0.995]
    assert mask_iou_mean(values) == pytest.approx(sum(values) / 3)
    assert mask_iou_min(values) == pytest.approx(0.985)


def test_depth_pearson_perfect_linear():
    x = np.arange(100, dtype=np.float64)
    y = 2.0 * x + 5.0
    assert depth_pearson_r(x.reshape(10, 10), y.reshape(10, 10)) == pytest.approx(1.0)


def test_depth_spearman_monotonic_nonlinear():
    x = np.arange(64, dtype=np.float64)
    y = x**2
    assert depth_spearman_r(x.reshape(8, 8), y.reshape(8, 8)) == pytest.approx(1.0)


def test_quality_gates_pass():
    result = evaluate_quality_gates(0.99, [0.96, 0.97, 0.98])
    assert result["q2_pass"] is True
    assert result["q3_pass"] is True
    assert result["pass"] is True


def test_quality_gates_fail_q2():
    result = evaluate_quality_gates(0.97, [0.96, 0.97])
    assert result["q2_pass"] is False
    assert result["pass"] is False


def test_quality_gate_constants_match_spec():
    assert Q2_GATE_MIN == 0.98
    assert Q3_GATE_MEDIAN_MIN == 0.95
