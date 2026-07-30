"""Checkpoint-one baselines with a shared probability interface."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from footcast.evaluation.metrics import CLASS_LABELS
from footcast.features.elo import EloConfig, expected_home_score


class MajorityClassBaseline:
    """Predict the most common training outcome and its empirical prior."""

    def fit(
        self, features: pd.DataFrame, target: Sequence[str]
    ) -> MajorityClassBaseline:
        del features
        values = pd.Series(target, dtype="object")
        if values.empty:
            raise ValueError("Majority baseline requires training targets")
        counts = values.value_counts().reindex(CLASS_LABELS, fill_value=0)
        self.class_probabilities_ = (counts / counts.sum()).to_numpy(
            dtype=float
        )
        self.majority_class_ = CLASS_LABELS[
            int(np.argmax(self.class_probabilities_))
        ]
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return np.full(len(features), self.majority_class_, dtype=object)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return np.tile(self.class_probabilities_, (len(features), 1))

    def _check_fitted(self) -> None:
        if not hasattr(self, "class_probabilities_"):
            raise RuntimeError("Fit the majority baseline before prediction")


class AlwaysHomeBaseline:
    """Predict a home win with certainty for every match."""

    def fit(
        self, features: pd.DataFrame, target: Sequence[str]
    ) -> AlwaysHomeBaseline:
        del features, target
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), "home_win", dtype=object)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probabilities = np.zeros((len(features), len(CLASS_LABELS)))
        probabilities[:, CLASS_LABELS.index("home_win")] = 1.0
        return probabilities


class EloBaseline:
    """Turn pre-match Elo expected scores into three outcome probabilities."""

    def __init__(self, config: EloConfig | None = None) -> None:
        self.config = config or EloConfig()

    def fit(
        self, features: pd.DataFrame, target: Sequence[str]
    ) -> EloBaseline:
        del features
        values = pd.Series(target, dtype="object")
        if values.empty:
            raise ValueError("Elo baseline requires training targets")
        self.draw_probability_ = float((values == "draw").mean())
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "draw_probability_"):
            raise RuntimeError("Fit the Elo baseline before prediction")
        required = {"home_elo", "away_elo"}
        missing = sorted(required - set(features.columns))
        if missing:
            raise ValueError(f"Elo baseline is missing columns: {missing}")

        home_scores = np.fromiter(
            (
                expected_home_score(home, away, self.config)
                for home, away in zip(
                    features["home_elo"],
                    features["away_elo"],
                    strict=True,
                )
            ),
            dtype=float,
            count=len(features),
        )
        decisive_probability = 1.0 - self.draw_probability_
        return np.column_stack(
            (
                decisive_probability * home_scores,
                np.full(len(features), self.draw_probability_),
                decisive_probability * (1.0 - home_scores),
            )
        )

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        indices = np.argmax(self.predict_proba(features), axis=1)
        return np.asarray(CLASS_LABELS, dtype=object)[indices]


class LogisticRegressionBaseline:
    """Median-imputed, scaled multinomial logistic regression."""

    def __init__(self) -> None:
        self.pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median", add_indicator=True),
                ),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2_000,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )

    def fit(
        self, features: pd.DataFrame, target: Sequence[str]
    ) -> LogisticRegressionBaseline:
        self.pipeline.fit(features, target)
        classes = set(self.pipeline.named_steps["classifier"].classes_)
        if classes != set(CLASS_LABELS):
            raise ValueError(
                "Training targets must contain all three outcome classes"
            )
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(features)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        raw = self.pipeline.predict_proba(features)
        classes = self.pipeline.named_steps["classifier"].classes_
        positions = [list(classes).index(label) for label in CLASS_LABELS]
        return raw[:, positions]
