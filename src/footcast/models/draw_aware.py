"""Two-stage classifier for draw and decisive-match probabilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from footcast.evaluation.metrics import CLASS_LABELS, validate_probabilities


def combine_two_stage_probabilities(
    draw_probabilities: np.ndarray,
    conditional_home_probabilities: np.ndarray,
) -> np.ndarray:
    """Combine P(draw) and P(home | decisive) into fixed-order outcomes."""
    draw = np.asarray(draw_probabilities, dtype=float)
    home_given_decisive = np.asarray(
        conditional_home_probabilities,
        dtype=float,
    )
    if draw.ndim != 1 or draw.shape != home_given_decisive.shape:
        raise ValueError("Two-stage probabilities require equal vectors")
    if len(draw) == 0:
        raise ValueError("Two-stage probabilities require at least one row")
    if not np.isfinite(draw).all() or not np.isfinite(home_given_decisive).all():
        raise ValueError("Two-stage probabilities must be finite")
    if (
        (draw < 0).any()
        or (draw > 1).any()
        or (home_given_decisive < 0).any()
        or (home_given_decisive > 1).any()
    ):
        raise ValueError("Two-stage probabilities must be between zero and one")

    decisive = 1.0 - draw
    combined = np.column_stack(
        (
            decisive * home_given_decisive,
            draw,
            decisive * (1.0 - home_given_decisive),
        )
    )
    return validate_probabilities(combined, row_count=len(draw))


def _binary_pipeline(*, positive_weight: float) -> Pipeline:
    if positive_weight < 1.0:
        raise ValueError("Positive-class weight must be at least 1.0")
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True),
            ),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight={0: 1.0, 1: positive_weight},
                    max_iter=2_000,
                    random_state=42,
                    solver="lbfgs",
                ),
            ),
        ]
    )


class TwoStageDrawClassifier:
    """Estimate draw first, then home versus away conditional on no draw."""

    def __init__(self, *, draw_weight: float = 1.0) -> None:
        if draw_weight < 1.0:
            raise ValueError("Draw weight must be at least 1.0")
        self.draw_weight = float(draw_weight)
        self.draw_pipeline = _binary_pipeline(positive_weight=self.draw_weight)
        self.decisive_pipeline = _binary_pipeline(positive_weight=1.0)

    def fit(
        self,
        features: pd.DataFrame,
        target: Sequence[str],
    ) -> TwoStageDrawClassifier:
        outcomes = pd.Series(target, dtype="object", index=features.index)
        if len(features) == 0 or len(features) != len(outcomes):
            raise ValueError("Features and targets require equal nonzero rows")
        unknown = set(outcomes) - set(CLASS_LABELS)
        if unknown:
            raise ValueError(f"Unknown outcome labels: {sorted(unknown)}")
        if set(outcomes) != set(CLASS_LABELS):
            raise ValueError("Training targets must contain all three outcomes")

        draw_target = (outcomes == "draw").astype(int)
        decisive_mask = outcomes != "draw"
        decisive_target = (outcomes.loc[decisive_mask] == "home_win").astype(int)
        self.draw_pipeline.fit(features, draw_target)
        self.decisive_pipeline.fit(
            features.loc[decisive_mask],
            decisive_target,
        )
        return self

    @staticmethod
    def _positive_probability(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
        classifier = pipeline.named_steps["classifier"]
        classes = list(classifier.classes_)
        return pipeline.predict_proba(features)[:, classes.index(1)]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        draw = self._positive_probability(self.draw_pipeline, features)
        home_given_decisive = self._positive_probability(
            self.decisive_pipeline,
            features,
        )
        return combine_two_stage_probabilities(draw, home_given_decisive)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(features)
        return np.asarray(CLASS_LABELS, dtype=object)[
            np.argmax(probabilities, axis=1)
        ]
