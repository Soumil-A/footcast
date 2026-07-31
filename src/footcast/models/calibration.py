"""Leakage-safe calibration for FootCast probability forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from footcast.evaluation.metrics import (
    CLASS_LABELS,
    evaluate_predictions,
    validate_probabilities,
)
from footcast.models.random_forest import (
    SELECTED_PARAMETERS,
    aligned_probabilities,
    expanding_season_folds,
    make_random_forest,
)

CALIBRATION_METHODS = ("uncalibrated", "sigmoid", "isotonic")


class ProbabilityCalibrator(Protocol):
    """Minimal interface shared by calibration methods."""

    def fit(
        self,
        probabilities: np.ndarray,
        target: np.ndarray,
    ) -> ProbabilityCalibrator: ...

    def transform(self, probabilities: np.ndarray) -> np.ndarray: ...


class IdentityCalibrator:
    """Leave valid probabilities unchanged."""

    def fit(
        self,
        probabilities: np.ndarray,
        target: np.ndarray,
    ) -> IdentityCalibrator:
        validate_probabilities(probabilities, row_count=len(target))
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        return validate_probabilities(
            probabilities,
            row_count=len(probabilities),
        ).copy()


class SigmoidCalibrator:
    """Fit multinomial logistic calibration on log probabilities."""

    def __init__(self) -> None:
        self.model = LogisticRegression(
            max_iter=2_000,
            solver="lbfgs",
            random_state=42,
        )

    @staticmethod
    def _features(probabilities: np.ndarray) -> np.ndarray:
        values = validate_probabilities(
            probabilities,
            row_count=len(probabilities),
        )
        return np.log(np.clip(values, 1e-6, 1.0))

    def fit(
        self,
        probabilities: np.ndarray,
        target: np.ndarray,
    ) -> SigmoidCalibrator:
        self.model.fit(self._features(probabilities), target)
        if set(self.model.classes_) != set(CLASS_LABELS):
            raise ValueError("Calibration targets require all outcome classes")
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        raw = self.model.predict_proba(self._features(probabilities))
        positions = [
            list(self.model.classes_).index(label) for label in CLASS_LABELS
        ]
        return raw[:, positions]


class IsotonicCalibrator:
    """Fit one monotonic probability mapping per outcome, then normalize."""

    def __init__(self) -> None:
        self.models = [
            IsotonicRegression(out_of_bounds="clip")
            for _ in CLASS_LABELS
        ]

    def fit(
        self,
        probabilities: np.ndarray,
        target: np.ndarray,
    ) -> IsotonicCalibrator:
        values = validate_probabilities(
            probabilities,
            row_count=len(target),
        )
        for class_index, (label, model) in enumerate(
            zip(CLASS_LABELS, self.models, strict=True)
        ):
            model.fit(values[:, class_index], (target == label).astype(float))
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        values = validate_probabilities(
            probabilities,
            row_count=len(probabilities),
        )
        calibrated = np.column_stack(
            [
                model.predict(values[:, class_index])
                for class_index, model in enumerate(self.models)
            ]
        )
        calibrated = np.clip(calibrated, 0.0, 1.0)
        totals = calibrated.sum(axis=1, keepdims=True)
        zero_rows = totals[:, 0] == 0
        calibrated[zero_rows] = 1.0 / len(CLASS_LABELS)
        totals[zero_rows] = 1.0
        return calibrated / totals


def make_calibrator(method: str) -> ProbabilityCalibrator:
    """Construct one named calibration method."""
    calibrators = {
        "uncalibrated": IdentityCalibrator,
        "sigmoid": SigmoidCalibrator,
        "isotonic": IsotonicCalibrator,
    }
    try:
        return calibrators[method]()
    except KeyError as error:
        raise ValueError(f"Unsupported calibration method: {method}") from error


@dataclass(frozen=True)
class OutOfFoldPredictions:
    """One next-season probability block from an earlier-season forest."""

    validation_season: str
    train_seasons: tuple[str, ...]
    probabilities: np.ndarray
    target: np.ndarray


def generate_oof_predictions(
    training: pd.DataFrame,
    feature_columns: list[str],
) -> list[OutOfFoldPredictions]:
    """Generate next-season probabilities from forests fit only on the past."""
    features = training[feature_columns].reset_index(drop=True)
    target = training["result"].reset_index(drop=True)
    folds = expanding_season_folds(training)
    predictions = []
    for fold in folds:
        model = make_random_forest(SELECTED_PARAMETERS)
        model.fit(
            features.iloc[fold["train_indices"]],
            target.iloc[fold["train_indices"]],
        )
        probabilities = aligned_probabilities(
            model,
            features.iloc[fold["validation_indices"]],
        )
        predictions.append(
            OutOfFoldPredictions(
                validation_season=fold["validation_season"],
                train_seasons=tuple(fold["train_seasons"]),
                probabilities=probabilities,
                target=target.iloc[fold["validation_indices"]].to_numpy(
                    dtype=object
                ),
            )
        )
    return predictions


def evaluate_calibration_methods(
    folds: list[OutOfFoldPredictions],
) -> list[dict[str, object]]:
    """Compare methods forward-only on later OOF seasons."""
    if len(folds) < 2:
        raise ValueError("Calibration selection requires at least two OOF folds")
    candidates = []
    for method in CALIBRATION_METHODS:
        evaluations = []
        for evaluation_index in range(1, len(folds)):
            calibration_folds = folds[:evaluation_index]
            evaluation_fold = folds[evaluation_index]
            calibration_probabilities = np.vstack(
                [fold.probabilities for fold in calibration_folds]
            )
            calibration_target = np.concatenate(
                [fold.target for fold in calibration_folds]
            )
            calibrator = make_calibrator(method).fit(
                calibration_probabilities,
                calibration_target,
            )
            probabilities = calibrator.transform(
                evaluation_fold.probabilities
            )
            predictions = np.asarray(CLASS_LABELS, dtype=object)[
                np.argmax(probabilities, axis=1)
            ]
            metrics = evaluate_predictions(
                evaluation_fold.target,
                predictions,
                probabilities,
            )
            evaluations.append(
                {
                    "calibration_seasons": [
                        fold.validation_season for fold in calibration_folds
                    ],
                    "evaluation_season": evaluation_fold.validation_season,
                    "calibration_rows": int(len(calibration_target)),
                    "evaluation_rows": int(len(evaluation_fold.target)),
                    "metrics": metrics,
                }
            )
        candidates.append(
            {
                "method": method,
                "mean_log_loss": float(
                    np.mean(
                        [
                            evaluation["metrics"]["log_loss"]
                            for evaluation in evaluations
                        ]
                    )
                ),
                "mean_brier_score": float(
                    np.mean(
                        [
                            evaluation["metrics"]["multiclass_brier_score"]
                            for evaluation in evaluations
                        ]
                    )
                ),
                "mean_expected_calibration_error": float(
                    np.mean(
                        [
                            evaluation["metrics"][
                                "expected_calibration_error"
                            ]
                            for evaluation in evaluations
                        ]
                    )
                ),
                "mean_macro_f1": float(
                    np.mean(
                        [
                            evaluation["metrics"]["macro_f1"]
                            for evaluation in evaluations
                        ]
                    )
                ),
                "folds": evaluations,
            }
        )
    return candidates


def select_calibration_method(
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    """Select mean log loss first, then Brier score."""
    if not candidates:
        raise ValueError("Cannot select from an empty calibration comparison")
    return min(
        candidates,
        key=lambda candidate: (
            float(candidate["mean_log_loss"]),
            float(candidate["mean_brier_score"]),
            str(candidate["method"]),
        ),
    )
