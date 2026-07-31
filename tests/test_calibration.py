"""Tests for Phase 5 calibration and validation-only diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footcast.evaluation.calibration_plots import (
    plot_error_slices,
    plot_reliability_comparison,
)
from footcast.evaluation.error_analysis import build_error_analysis
from footcast.evaluation.metrics import (
    classwise_calibration_bins,
    evaluate_predictions,
    expected_calibration_error,
)
from footcast.models.calibration import (
    CALIBRATION_METHODS,
    OutOfFoldPredictions,
    evaluate_calibration_methods,
    make_calibrator,
    select_calibration_method,
)


def _calibration_data() -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(
        [
            [0.70, 0.20, 0.10],
            [0.20, 0.60, 0.20],
            [0.10, 0.20, 0.70],
            [0.55, 0.25, 0.20],
            [0.25, 0.45, 0.30],
            [0.20, 0.25, 0.55],
        ]
    )
    target = np.asarray(
        ["home_win", "draw", "away_win", "draw", "home_win", "away_win"],
        dtype=object,
    )
    return probabilities, target


def test_probability_metrics_are_zero_for_perfect_certainty() -> None:
    actual = ["home_win", "draw", "away_win"]
    metrics = evaluate_predictions(actual, actual, np.eye(3))

    assert metrics["multiclass_brier_score"] == 0.0
    assert metrics["expected_calibration_error"] == 0.0


def test_expected_calibration_error_matches_one_bin_example() -> None:
    error = expected_calibration_error(
        np.asarray([1.0, 0.0]),
        np.asarray([0.8, 0.8]),
        bin_count=1,
    )

    assert error == pytest.approx(0.3)


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_calibrators_return_valid_three_way_probabilities(method: str) -> None:
    probabilities, target = _calibration_data()
    calibrator = make_calibrator(method).fit(probabilities, target)

    transformed = calibrator.transform(probabilities)

    assert transformed.shape == probabilities.shape
    assert transformed.min() >= 0
    assert transformed.max() <= 1
    assert transformed.sum(axis=1) == pytest.approx(np.ones(len(target)))


def test_calibration_comparison_is_strictly_forward_only() -> None:
    probabilities, target = _calibration_data()
    folds = [
        OutOfFoldPredictions(
            validation_season=season,
            train_seasons=("earlier",),
            probabilities=np.roll(probabilities, index, axis=0),
            target=target,
        )
        for index, season in enumerate(("2018-19", "2019-20", "2020-21"))
    ]

    candidates = evaluate_calibration_methods(folds)

    assert {candidate["method"] for candidate in candidates} == set(
        CALIBRATION_METHODS
    )
    for candidate in candidates:
        assert len(candidate["folds"]) == 2
        assert candidate["folds"][0]["calibration_seasons"] == ["2018-19"]
        assert candidate["folds"][0]["evaluation_season"] == "2019-20"
        assert candidate["folds"][1]["calibration_seasons"] == [
            "2018-19",
            "2019-20",
        ]
        assert candidate["folds"][1]["evaluation_season"] == "2020-21"


def test_calibration_selection_uses_log_loss_then_brier() -> None:
    candidates = [
        {
            "method": "worse_log_loss",
            "mean_log_loss": 1.0,
            "mean_brier_score": 0.4,
        },
        {
            "method": "worse_brier",
            "mean_log_loss": 0.9,
            "mean_brier_score": 0.5,
        },
        {
            "method": "selected",
            "mean_log_loss": 0.9,
            "mean_brier_score": 0.3,
        },
    ]

    assert select_calibration_method(candidates)["method"] == "selected"


def _validation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": ["2023-24"] * 6,
            "match_date": pd.date_range("2023-08-01", periods=6).astype(str),
            "home_team": [f"Home {index}" for index in range(6)],
            "away_team": [f"Away {index}" for index in range(6)],
            "result": [
                "home_win",
                "draw",
                "away_win",
                "home_win",
                "draw",
                "away_win",
            ],
            "home_history_matches": [0, 10, 10, 10, 10, 10],
            "away_history_matches": [10] * 6,
            "home_season_matches_played": [0, 2, 6, 7, 8, 9],
            "away_season_matches_played": [0, 2, 6, 7, 8, 9],
            "elo_difference": [0, 40, 75, 120, -130, -20],
        }
    )


def test_error_analysis_preserves_rows_and_identifies_slices() -> None:
    probabilities, _ = _calibration_data()
    analysis = build_error_analysis(_validation_frame(), probabilities)

    assert analysis["overall"]["rows"] == 6
    assert analysis["by_history"]["cold_start"]["rows"] == 1
    assert analysis["by_season_timing"]["early_season"]["rows"] == 2
    assert sum(
        group["rows"] for group in analysis["by_elo_gap"].values()
    ) == 6
    assert analysis["high_confidence_mistake_count"] >= 0


def test_calibration_bins_and_plots_are_written(tmp_path) -> None:
    probabilities, target = _calibration_data()
    bins = classwise_calibration_bins(target, probabilities, bin_count=4)
    analysis = build_error_analysis(_validation_frame(), probabilities)
    reliability_path = tmp_path / "reliability.png"
    errors_path = tmp_path / "errors.png"

    plot_reliability_comparison(bins, bins, reliability_path)
    plot_error_slices(analysis, errors_path)

    assert set(bins) == {"home_win", "draw", "away_win"}
    assert reliability_path.stat().st_size > 0
    assert errors_path.stat().st_size > 0
