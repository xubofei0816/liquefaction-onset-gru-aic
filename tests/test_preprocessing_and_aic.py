from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liquefaction_timing.workflow import (  # noqa: E402
    HARD_COUNT_DETECTION_THRESHOLD,
    MotionRecord,
    accumulated_aic_curves,
    analyze_aic,
    build_feature_variations,
    rotate_horizontal_components,
)


def test_rotation_rule_at_90_degrees() -> None:
    h1 = np.asarray([1.0, 2.0], dtype=np.float32)
    h2 = np.asarray([3.0, 4.0], dtype=np.float32)
    rotated_h1, rotated_h2 = rotate_horizontal_components(h1, h2, 90.0)
    np.testing.assert_allclose(rotated_h1, -h2, atol=1e-6)
    np.testing.assert_allclose(rotated_h2, h1, atol=1e-6)


def test_feature_order_and_variation_count() -> None:
    sample_count = 1000
    record = MotionRecord(
        recording_id="unit_test",
        time_s=np.arange(sample_count, dtype=np.float32) * 0.005,
        h1_g=np.linspace(-1.0, 1.0, sample_count, dtype=np.float32),
        h2_g=np.linspace(1.0, -1.0, sample_count, dtype=np.float32),
        v_g=np.zeros(sample_count, dtype=np.float32),
        mw=6.9,
    )
    variations = build_feature_variations(record)
    assert len(variations) == 18
    assert variations[0].shape == (sample_count, 6)
    np.testing.assert_allclose(variations[0][:, 0], record.h1_g)
    np.testing.assert_allclose(variations[0][:, 1], record.h2_g)
    np.testing.assert_allclose(variations[0][:, 2], record.v_g)


def test_probability_curve_uses_only_class1_argmax_windows() -> None:
    predictions = np.asarray(
        [
            [
                [0.55, 0.40, 0.05],  # Class 0: contributes zero.
                [0.10, 0.80, 0.10],  # Class 1: contributes 0.80.
            ]
        ],
        dtype=np.float64,
    )
    count, probability = accumulated_aic_curves(predictions, segment_points=2)
    np.testing.assert_allclose(count[0], [0.0, 1.0, 1.0])
    np.testing.assert_allclose(probability[0], [0.0, 0.8, 0.8])


def test_detection_threshold_and_probability_plateau_midpoint() -> None:
    window_count = 150
    predictions = np.zeros((18, window_count, 3), dtype=np.float64)
    predictions[:, :, 1] = 0.8
    predictions[:, :, 0] = 0.1
    predictions[:, :, 2] = 0.1
    result = analyze_aic(predictions)
    assert result.hard_count_peak == 150.0
    assert result.hard_count_peak >= HARD_COUNT_DETECTION_THRESHOLD
    assert result.detected
    assert result.predicted_tl_sec is not None


def test_no_tl_when_hard_count_peak_is_below_threshold() -> None:
    predictions = np.zeros((18, 50, 3), dtype=np.float64)
    predictions[:, :, 1] = 0.8
    predictions[:, :, 0] = 0.1
    predictions[:, :, 2] = 0.1
    result = analyze_aic(predictions, threshold=100.0)
    assert result.hard_count_peak == 50.0
    assert not result.detected
    assert result.predicted_tl_sec is None

