"""Tests for Phase 6 Poisson, Dixon-Coles, and rolling boundaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footcast.config import DATA_SPLIT
from footcast.evaluation.goal_model_plots import (
    plot_draw_recall,
    plot_goal_model_comparison,
    plot_log_loss_by_season,
)
from footcast.models.goal_models import (
    PoissonGoalModel,
    score_rates_to_outcome_probabilities,
    team_goal_rows,
)
from footcast.models.run_goal_models import V2_SEASONS, rolling_season_folds


def _matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "home_team": ["A", "B", "C", "A", "C", "B"],
            "away_team": ["B", "C", "A", "C", "B", "A"],
            "full_time_home_goals": [2, 1, 0, 3, 1, 2],
            "full_time_away_goals": [0, 1, 2, 1, 0, 2],
        }
    )


def test_team_goal_rows_preserve_both_team_perspectives() -> None:
    rows = team_goal_rows(_matches().iloc[[0]])

    assert rows.to_dict(orient="records") == [
        {"team": "A", "opponent": "B", "is_home": 1.0, "goals": 2},
        {"team": "B", "opponent": "A", "is_home": 0.0, "goals": 0},
    ]


def test_score_probabilities_are_valid_and_reflect_goal_rates() -> None:
    probabilities = score_rates_to_outcome_probabilities(
        np.asarray([2.0, 0.8]),
        np.asarray([0.8, 2.0]),
    )

    assert probabilities.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert probabilities[0, 0] > probabilities[0, 2]
    assert probabilities[1, 2] > probabilities[1, 0]


def test_negative_dixon_coles_rho_increases_low_score_draw_mass() -> None:
    independent = score_rates_to_outcome_probabilities(
        np.asarray([1.1]),
        np.asarray([1.0]),
        rho=0.0,
    )
    adjusted = score_rates_to_outcome_probabilities(
        np.asarray([1.1]),
        np.asarray([1.0]),
        rho=-0.1,
    )

    assert adjusted[0, 1] > independent[0, 1]


def test_poisson_model_predicts_for_seen_and_unseen_teams() -> None:
    model = PoissonGoalModel(alpha=0.5).fit(_matches())
    fixtures = pd.DataFrame(
        {
            "home_team": ["A", "Promoted"],
            "away_team": ["B", "A"],
        }
    )

    home_rates, away_rates = model.expected_goals(fixtures)
    probabilities = model.predict_proba(fixtures, rho=-0.1)

    assert (home_rates > 0).all()
    assert (away_rates > 0).all()
    assert probabilities.shape == (2, 3)
    assert probabilities.sum(axis=1) == pytest.approx([1.0, 1.0])


def _season_frame() -> pd.DataFrame:
    rows = []
    for season in V2_SEASONS:
        split = (
            "train"
            if season in DATA_SPLIT.train
            else "validation"
            if season in DATA_SPLIT.validation
            else "test"
        )
        rows.append({"season": season, "split": split})
    return pd.DataFrame(rows)


def test_rolling_folds_expand_through_seen_test_season() -> None:
    folds = rolling_season_folds(_season_frame())

    assert len(folds) == 7
    assert folds[0]["train_seasons"] == list(V2_SEASONS[:3])
    assert folds[0]["evaluation_season"] == "2018-19"
    assert folds[-1]["evaluation_season"] == "2024-25"
    for fold in folds:
        evaluation_index = V2_SEASONS.index(fold["evaluation_season"])
        assert fold["train_seasons"] == list(V2_SEASONS[:evaluation_index])


def test_rolling_folds_reject_holdout() -> None:
    frame = _season_frame()
    leaked = pd.DataFrame(
        [{"season": DATA_SPLIT.holdout[0], "split": "holdout"}]
    )

    with pytest.raises(ValueError, match="Holdout"):
        rolling_season_folds(pd.concat([frame, leaked], ignore_index=True))


def test_goal_model_plots_are_written(tmp_path) -> None:
    folds = [
        {
            "evaluation_season": "2023-24",
            "metrics": {"log_loss": 1.0},
        },
        {
            "evaluation_season": "2024-25",
            "metrics": {"log_loss": 0.95},
        },
    ]
    models = {
        "Elo": {
            "mean_accuracy": 0.52,
            "mean_macro_f1": 0.39,
            "mean_log_loss": 0.98,
            "mean_draw_recall": 0.0,
            "folds": folds,
        },
        "Dixon-Coles": {
            "mean_accuracy": 0.50,
            "mean_macro_f1": 0.45,
            "mean_log_loss": 0.96,
            "mean_draw_recall": 0.25,
            "folds": folds,
        },
    }
    comparison = tmp_path / "comparison.png"
    seasons = tmp_path / "seasons.png"
    draws = tmp_path / "draws.png"

    plot_goal_model_comparison(models, comparison)
    plot_log_loss_by_season(models, seasons)
    plot_draw_recall(models, draws)

    assert comparison.stat().st_size > 0
    assert seasons.stat().st_size > 0
    assert draws.stat().st_size > 0
