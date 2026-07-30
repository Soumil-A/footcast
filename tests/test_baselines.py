"""Tests for Phase 4 baseline models and chronological safeguards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footcast.config import DATA_SPLIT
from footcast.evaluation.metrics import CLASS_LABELS, evaluate_predictions
from footcast.evaluation.plots import plot_confusion_matrices
from footcast.models.baselines import (
    AlwaysHomeBaseline,
    EloBaseline,
    LogisticRegressionBaseline,
    MajorityClassBaseline,
)
from footcast.models.run_baselines import split_development_features


def test_majority_baseline_uses_training_frequencies() -> None:
    features = pd.DataFrame({"signal": [0, 1, 2]})
    model = MajorityClassBaseline().fit(
        features,
        ["home_win", "home_win", "draw"],
    )

    validation = pd.DataFrame({"signal": [100, 200]})
    assert model.predict(validation).tolist() == ["home_win", "home_win"]
    np.testing.assert_allclose(
        model.predict_proba(validation),
        [[2 / 3, 1 / 3, 0], [2 / 3, 1 / 3, 0]],
    )


def test_always_home_baseline_is_deterministic() -> None:
    features = pd.DataFrame({"signal": [1, 2]})
    model = AlwaysHomeBaseline().fit(features, ["draw", "away_win"])

    assert model.predict(features).tolist() == ["home_win", "home_win"]
    assert model.predict_proba(features).tolist() == [[1, 0, 0], [1, 0, 0]]


def test_elo_probabilities_are_valid_and_respond_to_rating_gap() -> None:
    training = pd.DataFrame({"home_elo": [1500] * 4, "away_elo": [1500] * 4})
    model = EloBaseline().fit(
        training,
        ["home_win", "draw", "away_win", "home_win"],
    )
    matches = pd.DataFrame(
        {
            "home_elo": [1700, 1300],
            "away_elo": [1300, 1700],
        }
    )

    probabilities = model.predict_proba(matches)
    assert probabilities.sum(axis=1) == pytest.approx([1, 1])
    assert probabilities[0, 0] > probabilities[0, 2]
    assert probabilities[1, 2] > probabilities[1, 0]
    assert probabilities[:, 1] == pytest.approx([0.25, 0.25])


def test_logistic_pipeline_fits_imputation_on_training_only() -> None:
    training = pd.DataFrame(
        {
            "form": [0.0, 2.0, 4.0, np.nan, 8.0, 10.0],
            "elo": [1400, 1450, 1500, 1550, 1600, 1650],
        }
    )
    target = [
        "away_win",
        "draw",
        "home_win",
        "away_win",
        "draw",
        "home_win",
    ]
    model = LogisticRegressionBaseline().fit(training, target)

    assert model.pipeline.named_steps["imputer"].statistics_[0] == 4.0
    validation = pd.DataFrame({"form": [1000.0, np.nan], "elo": [1700, 1350]})
    probabilities = model.predict_proba(validation)
    assert probabilities.shape == (2, 3)
    assert probabilities.sum(axis=1) == pytest.approx([1, 1])


def _development_frame() -> pd.DataFrame:
    rows = []
    for index, season in enumerate(DATA_SPLIT.train):
        rows.append(
            {
                "season": season,
                "split": "train",
                "match_date": f"{2015 + index}-08-01",
                "home_team": f"Home {index}",
                "away_team": f"Away {index}",
                "signal": float(index),
                "result": CLASS_LABELS[index % 3],
            }
        )
    rows.append(
        {
            "season": DATA_SPLIT.validation[0],
            "split": "validation",
            "match_date": "2023-08-01",
            "home_team": "Validation home",
            "away_team": "Validation away",
            "signal": 9.0,
            "result": "draw",
        }
    )
    return pd.DataFrame(rows)


def test_split_development_features_enforces_chronology() -> None:
    training, validation = split_development_features(_development_frame())

    assert set(training["season"]) == set(DATA_SPLIT.train)
    assert validation["season"].tolist() == list(DATA_SPLIT.validation)
    assert training["match_date"].max() < validation["match_date"].min()


@pytest.mark.parametrize("forbidden_split", ["test", "holdout"])
def test_split_development_features_rejects_reserved_data(
    forbidden_split: str,
) -> None:
    features = _development_frame()
    leaked = features.iloc[[0]].copy()
    leaked["split"] = forbidden_split
    leaked["season"] = (
        DATA_SPLIT.test[0]
        if forbidden_split == "test"
        else DATA_SPLIT.holdout[0]
    )

    with pytest.raises(ValueError, match="Test or holdout"):
        split_development_features(pd.concat([features, leaked]))


def test_metrics_match_hand_calculation() -> None:
    actual = ["home_win", "draw", "away_win"]
    predicted = ["home_win", "home_win", "away_win"]
    probabilities = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.6, 0.3, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )

    metrics = evaluate_predictions(actual, predicted, probabilities)

    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["per_class_recall"] == {
        "home_win": 1.0,
        "draw": 0.0,
        "away_win": 1.0,
    }
    assert metrics["confusion_matrix"] == [[1, 0, 0], [1, 0, 0], [0, 0, 1]]


def test_metrics_reject_invalid_probability_rows() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        evaluate_predictions(
            ["home_win"],
            ["home_win"],
            np.asarray([[0.8, 0.1, 0.0]]),
        )


def test_confusion_matrix_plot_is_written(tmp_path) -> None:
    metrics = evaluate_predictions(
        ["home_win", "draw", "away_win"],
        ["home_win", "draw", "away_win"],
        np.eye(3),
    )
    results = {name: metrics for name in (
        "Majority class",
        "Always home",
        "Elo",
        "Logistic regression",
    )}
    destination = tmp_path / "confusion.png"

    plot_confusion_matrices(results, destination)

    assert destination.stat().st_size > 0
