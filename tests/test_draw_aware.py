"""Tests for Phase 6 draw-aware modeling and the fixed decision gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footcast.evaluation.draw_aware_plots import (
    plot_draw_aware_by_season,
    plot_draw_aware_comparison,
    plot_draw_weight_tradeoff,
)
from footcast.models.draw_aware import (
    TwoStageDrawClassifier,
    combine_two_stage_probabilities,
)
from footcast.models.run_draw_aware import (
    ACCEPTANCE_THRESHOLDS,
    checkpoint_decision,
    select_draw_candidate,
)
from footcast.models.run_goal_models import V2_DEVELOPMENT_SPLITS


def _training_data() -> tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame(
        {
            "elo_difference": [
                180,
                -160,
                5,
                140,
                -120,
                -8,
                90,
                -75,
                2,
                60,
                -50,
                0,
            ],
            "form_difference": [
                6,
                -5,
                0,
                5,
                -4,
                1,
                3,
                -3,
                0,
                2,
                -2,
                0,
            ],
        }
    )
    target = pd.Series(
        [
            "home_win",
            "away_win",
            "draw",
            "home_win",
            "away_win",
            "draw",
            "home_win",
            "away_win",
            "draw",
            "home_win",
            "away_win",
            "draw",
        ]
    )
    return features, target


def test_two_stage_composition_matches_hand_calculation() -> None:
    probabilities = combine_two_stage_probabilities(
        np.asarray([0.25, 0.40]),
        np.asarray([0.60, 0.20]),
    )

    np.testing.assert_allclose(
        probabilities,
        np.asarray(
            [
            [0.45, 0.25, 0.30],
            [0.12, 0.40, 0.48],
            ]
        ),
    )


def test_two_stage_composition_rejects_invalid_vectors() -> None:
    with pytest.raises(ValueError, match="equal vectors"):
        combine_two_stage_probabilities(np.asarray([0.2]), np.asarray([0.4, 0.5]))
    with pytest.raises(ValueError, match="between zero and one"):
        combine_two_stage_probabilities(np.asarray([1.2]), np.asarray([0.5]))


def test_two_stage_model_returns_valid_fixed_order_probabilities() -> None:
    features, target = _training_data()
    model = TwoStageDrawClassifier(draw_weight=1.5).fit(features, target)

    probabilities = model.predict_proba(features.iloc[:3])

    assert probabilities.shape == (3, 3)
    assert probabilities.sum(axis=1) == pytest.approx([1.0, 1.0, 1.0])
    assert model.predict(features.iloc[:3]).tolist() == [
        "home_win",
        "away_win",
        "draw",
    ]


def test_two_stage_model_rejects_missing_outcome_class() -> None:
    features, target = _training_data()
    target = target.replace("draw", "home_win")

    with pytest.raises(ValueError, match="all three outcomes"):
        TwoStageDrawClassifier().fit(features, target)


def test_candidate_selection_is_deterministic() -> None:
    candidates = [
        {
            "parameters": {"draw_weight": 1.5},
            "mean_log_loss": 0.98,
            "mean_macro_f1": 0.42,
        },
        {
            "parameters": {"draw_weight": 1.0},
            "mean_log_loss": 0.98,
            "mean_macro_f1": 0.42,
        },
        {
            "parameters": {"draw_weight": 2.0},
            "mean_log_loss": 0.99,
            "mean_macro_f1": 0.50,
        },
    ]

    selected = select_draw_candidate(candidates)

    assert selected["parameters"] == {"draw_weight": 1.0}


def test_checkpoint_gate_passes_exact_boundaries() -> None:
    decision = checkpoint_decision(
        {
            "mean_log_loss": ACCEPTANCE_THRESHOLDS["maximum_mean_log_loss"],
            "mean_multiclass_brier_score": ACCEPTANCE_THRESHOLDS[
                "maximum_mean_brier_score"
            ],
            "mean_macro_f1": ACCEPTANCE_THRESHOLDS["minimum_mean_macro_f1"],
            "mean_draw_recall": ACCEPTANCE_THRESHOLDS[
                "minimum_mean_draw_recall"
            ],
        }
    )

    assert decision["promoted"] is True
    assert decision["failed_checks"] == []
    assert decision["phase7_reference_model"].startswith("Two-stage")


def test_checkpoint_gate_rejects_when_any_criterion_fails() -> None:
    decision = checkpoint_decision(
        {
            "mean_log_loss": 0.971,
            "mean_multiclass_brier_score": 0.578,
            "mean_macro_f1": 0.400,
            "mean_draw_recall": 0.100,
        }
    )

    assert decision["promoted"] is False
    assert decision["failed_checks"] == ["mean_log_loss_at_most_0.970"]
    assert decision["phase7_reference_model"] == "Elo"


def test_draw_aware_development_splits_exclude_holdout() -> None:
    assert V2_DEVELOPMENT_SPLITS == {"train", "validation", "test"}
    assert "holdout" not in V2_DEVELOPMENT_SPLITS


def test_draw_aware_plots_are_written(tmp_path) -> None:
    folds = [
        {"evaluation_season": "2023-24", "metrics": {"log_loss": 0.98}},
        {"evaluation_season": "2024-25", "metrics": {"log_loss": 1.00}},
    ]
    models = {
        "Elo": {
            "mean_log_loss": 0.99,
            "mean_multiclass_brier_score": 0.59,
            "mean_macro_f1": 0.40,
            "mean_draw_recall": 0.0,
            "folds": folds,
        },
        "Two-stage": {
            "mean_log_loss": 0.97,
            "mean_multiclass_brier_score": 0.57,
            "mean_macro_f1": 0.42,
            "mean_draw_recall": 0.12,
            "folds": folds,
        },
    }
    comparison = tmp_path / "comparison.png"
    seasons = tmp_path / "seasons.png"
    tradeoff = tmp_path / "tradeoff.png"
    candidates = [
        {
            "parameters": {"draw_weight": 1.0},
            "mean_log_loss": 0.98,
            "mean_draw_recall": 0.02,
        },
        {
            "parameters": {"draw_weight": 2.0},
            "mean_log_loss": 1.04,
            "mean_draw_recall": 0.34,
        },
    ]

    plot_draw_aware_comparison(models, comparison)
    plot_draw_aware_by_season(models, seasons)
    plot_draw_weight_tradeoff(candidates, tradeoff)

    assert comparison.stat().st_size > 0
    assert seasons.stat().st_size > 0
    assert tradeoff.stat().st_size > 0
