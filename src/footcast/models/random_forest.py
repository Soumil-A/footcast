"""Random Forest model and time-aware selection utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from footcast.config import DATA_SPLIT
from footcast.evaluation.metrics import CLASS_LABELS, evaluate_predictions

RANDOM_STATE = 42
N_ESTIMATORS = 300
PARAMETER_GRID = (
    {"max_depth": 6, "min_samples_leaf": 5, "class_weight": None},
    {
        "max_depth": 6,
        "min_samples_leaf": 5,
        "class_weight": "balanced_subsample",
    },
    {"max_depth": 6, "min_samples_leaf": 20, "class_weight": None},
    {
        "max_depth": 6,
        "min_samples_leaf": 20,
        "class_weight": "balanced_subsample",
    },
    {"max_depth": 12, "min_samples_leaf": 5, "class_weight": None},
    {
        "max_depth": 12,
        "min_samples_leaf": 5,
        "class_weight": "balanced_subsample",
    },
    {"max_depth": 12, "min_samples_leaf": 20, "class_weight": None},
    {
        "max_depth": 12,
        "min_samples_leaf": 20,
        "class_weight": "balanced_subsample",
    },
    {"max_depth": None, "min_samples_leaf": 5, "class_weight": None},
    {
        "max_depth": None,
        "min_samples_leaf": 5,
        "class_weight": "balanced_subsample",
    },
    {"max_depth": None, "min_samples_leaf": 20, "class_weight": None},
    {
        "max_depth": None,
        "min_samples_leaf": 20,
        "class_weight": "balanced_subsample",
    },
)


def make_random_forest(params: dict[str, Any]) -> Pipeline:
    """Create a deterministic forest with training-fitted missing handling."""
    unknown = set(params) - {
        "max_depth",
        "min_samples_leaf",
        "class_weight",
    }
    if unknown:
        raise ValueError(f"Unsupported Random Forest settings: {sorted(unknown)}")
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    max_features="sqrt",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    **params,
                ),
            ),
        ]
    )


def aligned_probabilities(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return forest probabilities in FootCast's fixed class order."""
    classifier = model.named_steps["classifier"]
    check_is_fitted(classifier)
    raw = model.predict_proba(features)
    positions = [list(classifier.classes_).index(label) for label in CLASS_LABELS]
    return raw[:, positions]


def expanding_season_folds(
    training: pd.DataFrame,
    *,
    minimum_training_seasons: int = 3,
) -> list[dict[str, Any]]:
    """Build expanding folds where one later season validates earlier seasons."""
    if "season" not in training:
        raise ValueError("Training rows require a season column")
    observed = set(training["season"])
    expected = set(DATA_SPLIT.train)
    if observed != expected:
        raise ValueError("Cross-validation requires every frozen training season")
    if not 1 <= minimum_training_seasons < len(DATA_SPLIT.train):
        raise ValueError("minimum_training_seasons must leave a validation season")

    folds: list[dict[str, Any]] = []
    seasons = DATA_SPLIT.train
    for validation_position in range(minimum_training_seasons, len(seasons)):
        train_seasons = seasons[:validation_position]
        validation_season = seasons[validation_position]
        train_indices = np.flatnonzero(training["season"].isin(train_seasons))
        validation_indices = np.flatnonzero(
            training["season"] == validation_season
        )
        if len(train_indices) == 0 or len(validation_indices) == 0:
            raise ValueError("Every time-aware fold requires train and validation rows")
        folds.append(
            {
                "train_seasons": list(train_seasons),
                "validation_season": validation_season,
                "train_indices": train_indices,
                "validation_indices": validation_indices,
            }
        )
    return folds


def evaluate_parameter_grid(
    training: pd.DataFrame,
    feature_columns: list[str],
    *,
    parameter_grid: tuple[dict[str, Any], ...] = PARAMETER_GRID,
) -> list[dict[str, Any]]:
    """Evaluate each candidate only on expanding folds within training."""
    if not parameter_grid:
        raise ValueError("At least one Random Forest candidate is required")
    folds = expanding_season_folds(training)
    x = training[feature_columns].reset_index(drop=True)
    y = training["result"].reset_index(drop=True)
    candidates: list[dict[str, Any]] = []

    for params in parameter_grid:
        fold_results = []
        for fold in folds:
            train_indices = fold["train_indices"]
            validation_indices = fold["validation_indices"]
            model = make_random_forest(params)
            model.fit(x.iloc[train_indices], y.iloc[train_indices])
            predictions = model.predict(x.iloc[validation_indices])
            probabilities = aligned_probabilities(
                model,
                x.iloc[validation_indices],
            )
            metrics = evaluate_predictions(
                y.iloc[validation_indices],
                predictions,
                probabilities,
            )
            fold_results.append(
                {
                    "train_seasons": fold["train_seasons"],
                    "validation_season": fold["validation_season"],
                    "training_rows": int(len(train_indices)),
                    "validation_rows": int(len(validation_indices)),
                    "metrics": metrics,
                }
            )
        candidates.append(
            {
                "parameters": params,
                "mean_log_loss": float(
                    np.mean(
                        [
                            fold["metrics"]["log_loss"]
                            for fold in fold_results
                        ]
                    )
                ),
                "mean_macro_f1": float(
                    np.mean(
                        [
                            fold["metrics"]["macro_f1"]
                            for fold in fold_results
                        ]
                    )
                ),
                "folds": fold_results,
            }
        )
    return candidates


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose lowest mean log loss, breaking exact ties by macro F1."""
    if not candidates:
        raise ValueError("Cannot select from an empty candidate list")
    return min(
        candidates,
        key=lambda candidate: (
            candidate["mean_log_loss"],
            -candidate["mean_macro_f1"],
            str(candidate["parameters"]),
        ),
    )


def feature_importances(
    model: Pipeline,
    input_columns: list[str],
) -> list[dict[str, float | str]]:
    """Return fitted impurity importances, including missing indicators."""
    imputer = model.named_steps["imputer"]
    classifier = model.named_steps["classifier"]
    check_is_fitted(classifier)
    names = imputer.get_feature_names_out(input_columns)
    values = classifier.feature_importances_
    if len(names) != len(values):
        raise ValueError("Feature names and importance values do not align")
    return sorted(
        (
            {"feature": str(name), "importance": float(value)}
            for name, value in zip(names, values, strict=True)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
