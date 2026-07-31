"""Poisson and Dixon-Coles score-distribution models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from footcast.evaluation.metrics import CLASS_LABELS, validate_probabilities

REQUIRED_GOAL_COLUMNS = frozenset(
    {
        "home_team",
        "away_team",
        "full_time_home_goals",
        "full_time_away_goals",
    }
)


def team_goal_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """Convert fixtures into one attack/defense observation per team."""
    missing = sorted(REQUIRED_GOAL_COLUMNS - set(matches.columns))
    if missing:
        raise ValueError(f"Goal model is missing columns: {missing}")
    if matches.empty:
        raise ValueError("Goal model requires completed matches")
    home = pd.DataFrame(
        {
            "team": matches["home_team"].to_numpy(),
            "opponent": matches["away_team"].to_numpy(),
            "is_home": 1.0,
            "goals": matches["full_time_home_goals"].to_numpy(),
        }
    )
    away = pd.DataFrame(
        {
            "team": matches["away_team"].to_numpy(),
            "opponent": matches["home_team"].to_numpy(),
            "is_home": 0.0,
            "goals": matches["full_time_away_goals"].to_numpy(),
        }
    )
    rows = pd.concat([home, away], ignore_index=True)
    if rows["goals"].isna().any() or (rows["goals"] < 0).any():
        raise ValueError("Goal targets must be complete and nonnegative")
    return rows


def _design_pipeline(alpha: float) -> Pipeline:
    if alpha < 0:
        raise ValueError("Poisson regularization alpha cannot be negative")
    transformer = ColumnTransformer(
        transformers=[
            (
                "teams",
                OneHotEncoder(handle_unknown="ignore"),
                ["team", "opponent"],
            ),
            ("venue", "passthrough", ["is_home"]),
        ]
    )
    return Pipeline(
        steps=[
            ("design", transformer),
            (
                "poisson",
                PoissonRegressor(alpha=alpha, max_iter=1_000),
            ),
        ]
    )


def _fixture_rows(matches: pd.DataFrame, *, home: bool) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team": (
                matches["home_team"].to_numpy()
                if home
                else matches["away_team"].to_numpy()
            ),
            "opponent": (
                matches["away_team"].to_numpy()
                if home
                else matches["home_team"].to_numpy()
            ),
            "is_home": 1.0 if home else 0.0,
        }
    )


@dataclass
class PoissonGoalModel:
    """Regularized team-attack/opponent-defense Poisson regression."""

    alpha: float = 0.5

    def __post_init__(self) -> None:
        self.pipeline = _design_pipeline(self.alpha)

    def fit(self, matches: pd.DataFrame) -> PoissonGoalModel:
        rows = team_goal_rows(matches)
        self.pipeline.fit(rows[["team", "opponent", "is_home"]], rows["goals"])
        return self

    def expected_goals(self, matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict positive home and away scoring rates for each fixture."""
        missing = sorted({"home_team", "away_team"} - set(matches.columns))
        if missing:
            raise ValueError(f"Goal prediction is missing columns: {missing}")
        home = np.clip(
            self.pipeline.predict(_fixture_rows(matches, home=True)),
            1e-6,
            None,
        )
        away = np.clip(
            self.pipeline.predict(_fixture_rows(matches, home=False)),
            1e-6,
            None,
        )
        return home, away

    def predict_proba(
        self,
        matches: pd.DataFrame,
        *,
        rho: float = 0.0,
        max_goals: int = 10,
    ) -> np.ndarray:
        home, away = self.expected_goals(matches)
        return score_rates_to_outcome_probabilities(
            home,
            away,
            rho=rho,
            max_goals=max_goals,
        )

    def predict(self, matches: pd.DataFrame, *, rho: float = 0.0) -> np.ndarray:
        probabilities = self.predict_proba(matches, rho=rho)
        return np.asarray(CLASS_LABELS, dtype=object)[
            np.argmax(probabilities, axis=1)
        ]


def _poisson_probabilities(rate: float, max_goals: int) -> np.ndarray:
    values = np.empty(max_goals + 1, dtype=float)
    values[0] = np.exp(-rate)
    for goals in range(1, max_goals + 1):
        values[goals] = values[goals - 1] * rate / goals
    return values


def _dixon_coles_adjustment(
    home_goals: int,
    away_goals: int,
    home_rate: float,
    away_rate: float,
    rho: float,
) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1.0 - home_rate * away_rate * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + home_rate * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + away_rate * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def score_rates_to_outcome_probabilities(
    home_rates: np.ndarray,
    away_rates: np.ndarray,
    *,
    rho: float = 0.0,
    max_goals: int = 10,
) -> np.ndarray:
    """Convert expected goals to normalized home/draw/away probabilities."""
    home = np.asarray(home_rates, dtype=float)
    away = np.asarray(away_rates, dtype=float)
    if home.shape != away.shape or home.ndim != 1 or len(home) == 0:
        raise ValueError("Home and away rates require equal nonzero vectors")
    if not np.isfinite(home).all() or not np.isfinite(away).all():
        raise ValueError("Expected goal rates must be finite")
    if (home <= 0).any() or (away <= 0).any():
        raise ValueError("Expected goal rates must be positive")
    if not 3 <= max_goals <= 20:
        raise ValueError("max_goals must be between 3 and 20")
    if not -0.25 <= rho <= 0.25:
        raise ValueError("Dixon-Coles rho must be between -0.25 and 0.25")

    rows = []
    for home_rate, away_rate in zip(home, away, strict=True):
        score_matrix = np.outer(
            _poisson_probabilities(home_rate, max_goals),
            _poisson_probabilities(away_rate, max_goals),
        )
        for home_goals in (0, 1):
            for away_goals in (0, 1):
                score_matrix[home_goals, away_goals] *= (
                    _dixon_coles_adjustment(
                        home_goals,
                        away_goals,
                        home_rate,
                        away_rate,
                        rho,
                    )
                )
        score_matrix = np.clip(score_matrix, 0.0, None)
        score_matrix /= score_matrix.sum()
        rows.append(
            [
                float(np.tril(score_matrix, k=-1).sum()),
                float(np.trace(score_matrix)),
                float(np.triu(score_matrix, k=1).sum()),
            ]
        )
    return validate_probabilities(np.asarray(rows), row_count=len(home))
