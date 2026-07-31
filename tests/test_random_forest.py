"""Tests for time-aware Random Forest model selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footcast.config import DATA_SPLIT
from footcast.evaluation.metrics import evaluate_predictions
from footcast.evaluation.plots import (
    plot_feature_importances,
    plot_model_comparison,
    plot_single_confusion_matrix,
)
from footcast.models.random_forest import (
    aligned_probabilities,
    evaluate_parameter_grid,
    expanding_season_folds,
    feature_importances,
    make_random_forest,
    select_candidate,
)


def _training_seasons() -> pd.DataFrame:
    rows = []
    labels = ("home_win", "draw", "away_win")
    for season_index, season in enumerate(DATA_SPLIT.train):
        for label_index, label in enumerate(labels):
            rows.append(
                {
                    "season": season,
                    "signal": float(season_index + label_index),
                    "rest": (
                        np.nan
                        if season_index == 0 and label_index == 0
                        else float(label_index)
                    ),
                    "result": label,
                }
            )
    return pd.DataFrame(rows)


def test_expanding_folds_never_train_on_future_seasons() -> None:
    training = _training_seasons()
    folds = expanding_season_folds(training)

    assert len(folds) == 5
    assert folds[0]["train_seasons"] == list(DATA_SPLIT.train[:3])
    assert folds[0]["validation_season"] == DATA_SPLIT.train[3]
    assert folds[-1]["validation_season"] == DATA_SPLIT.train[-1]
    for fold in folds:
        validation_position = DATA_SPLIT.train.index(
            fold["validation_season"]
        )
        assert all(
            DATA_SPLIT.train.index(season) < validation_position
            for season in fold["train_seasons"]
        )
        assert not set(fold["train_indices"]) & set(
            fold["validation_indices"]
        )


def test_expanding_folds_reject_incomplete_training_period() -> None:
    incomplete = _training_seasons().query("season != '2015-16'")

    with pytest.raises(ValueError, match="every frozen training season"):
        expanding_season_folds(incomplete)


def test_random_forest_handles_missing_values_and_aligns_probabilities() -> None:
    training = _training_seasons()
    features = training[["signal", "rest"]]
    model = make_random_forest(
        {"max_depth": 6, "min_samples_leaf": 1, "class_weight": None}
    )
    model.fit(features, training["result"])

    assert model.named_steps["imputer"].statistics_[1] == pytest.approx(1.0)
    probabilities = aligned_probabilities(
        model,
        pd.DataFrame({"signal": [2.5], "rest": [np.nan]}),
    )
    assert probabilities.shape == (1, 3)
    assert probabilities.sum(axis=1) == pytest.approx([1.0])
    assert set(model.predict(features)) <= {
        "home_win",
        "draw",
        "away_win",
    }


def test_parameter_grid_records_every_expanding_fold() -> None:
    training = _training_seasons()
    candidates = evaluate_parameter_grid(
        training,
        ["signal", "rest"],
        parameter_grid=(
            {
                "max_depth": 4,
                "min_samples_leaf": 1,
                "class_weight": None,
            },
        ),
    )

    assert len(candidates) == 1
    assert len(candidates[0]["folds"]) == 5
    assert candidates[0]["mean_log_loss"] > 0
    assert 0 <= candidates[0]["mean_macro_f1"] <= 1


def test_candidate_selection_uses_log_loss_then_macro_f1() -> None:
    candidates = [
        {
            "parameters": {"name": "higher loss"},
            "mean_log_loss": 1.0,
            "mean_macro_f1": 0.9,
        },
        {
            "parameters": {"name": "tie lower F1"},
            "mean_log_loss": 0.8,
            "mean_macro_f1": 0.3,
        },
        {
            "parameters": {"name": "selected"},
            "mean_log_loss": 0.8,
            "mean_macro_f1": 0.5,
        },
    ]

    selected = select_candidate(candidates)

    assert selected["parameters"]["name"] == "selected"


def test_feature_importances_are_complete_and_sorted() -> None:
    training = _training_seasons()
    columns = ["signal", "rest"]
    model = make_random_forest(
        {"max_depth": 4, "min_samples_leaf": 1, "class_weight": None}
    )
    model.fit(training[columns], training["result"])

    importances = feature_importances(model, columns)

    assert sum(item["importance"] for item in importances) == pytest.approx(1.0)
    assert [item["importance"] for item in importances] == sorted(
        [item["importance"] for item in importances],
        reverse=True,
    )
    assert {"signal", "rest"} <= {
        str(item["feature"]) for item in importances
    }


def test_random_forest_plots_are_written(tmp_path) -> None:
    metrics = evaluate_predictions(
        ["home_win", "draw", "away_win"],
        ["home_win", "draw", "away_win"],
        np.eye(3),
    )
    confusion_path = tmp_path / "confusion.png"
    comparison_path = tmp_path / "comparison.png"
    importance_path = tmp_path / "importance.png"

    plot_single_confusion_matrix(metrics, confusion_path)
    plot_model_comparison(
        {"Elo": metrics, "Random Forest": metrics},
        comparison_path,
    )
    plot_feature_importances(
        [
            {"feature": "elo_difference", "importance": 0.7},
            {"feature": "form_points_difference", "importance": 0.3},
        ],
        importance_path,
    )

    assert confusion_path.stat().st_size > 0
    assert comparison_path.stat().st_size > 0
    assert importance_path.stat().st_size > 0
