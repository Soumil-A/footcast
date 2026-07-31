"""Frozen FootCast v1 model and final-evaluation contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib

from footcast.config import DATA_SPLIT
from footcast.evaluation.metrics import CLASS_LABELS
from footcast.models.random_forest import (
    N_ESTIMATORS,
    SELECTED_PARAMETERS,
    make_random_forest,
)

MODEL_VERSION = "footcast-rf-v1"
CALIBRATION_METHOD = "uncalibrated"
FROZEN_FEATURE_COLUMNS = (
    "home_history_matches",
    "home_rolling_matches",
    "home_form_points_last_5",
    "home_form_wins_last_5",
    "home_goals_for_last_5",
    "home_goals_against_last_5",
    "home_shots_last_5",
    "home_shots_on_target_last_5",
    "home_venue_matches_last_5",
    "home_venue_points_last_5",
    "home_days_since_previous_match",
    "home_season_matches_played",
    "home_season_points",
    "home_expanding_goals_for_mean",
    "home_expanding_goals_against_mean",
    "home_elo",
    "away_history_matches",
    "away_rolling_matches",
    "away_form_points_last_5",
    "away_form_wins_last_5",
    "away_goals_for_last_5",
    "away_goals_against_last_5",
    "away_shots_last_5",
    "away_shots_on_target_last_5",
    "away_venue_matches_last_5",
    "away_venue_points_last_5",
    "away_days_since_previous_match",
    "away_season_matches_played",
    "away_season_points",
    "away_expanding_goals_for_mean",
    "away_expanding_goals_against_mean",
    "away_elo",
    "elo_difference",
    "form_points_difference",
    "goals_scored_difference",
    "goals_conceded_difference",
    "rest_days_difference",
    "shots_on_target_difference",
)


def frozen_specification() -> dict[str, Any]:
    """Return the immutable, JSON-serializable v1 model contract."""
    return {
        "model_version": MODEL_VERSION,
        "target": list(CLASS_LABELS),
        "estimator": "sklearn.ensemble.RandomForestClassifier",
        "random_forest": {
            "n_estimators": N_ESTIMATORS,
            "max_features": "sqrt",
            "random_state": 42,
            **SELECTED_PARAMETERS,
        },
        "preprocessing": {
            "imputation": "training median",
            "missing_indicators": True,
            "scaling": False,
        },
        "calibration": CALIBRATION_METHOD,
        "feature_columns": list(FROZEN_FEATURE_COLUMNS),
        "fit_seasons": list(DATA_SPLIT.train + DATA_SPLIT.validation),
        "test_seasons": list(DATA_SPLIT.test),
        "holdout_seasons": list(DATA_SPLIT.holdout),
    }


def specification_sha256() -> str:
    """Hash the canonical frozen contract for artifact/report linkage."""
    encoded = json.dumps(
        frozen_specification(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fit_frozen_model(features, target):
    """Fit only the already-selected v1 estimator."""
    model = make_random_forest(SELECTED_PARAMETERS)
    model.fit(features[list(FROZEN_FEATURE_COLUMNS)], target)
    return model


def save_frozen_artifact(model, destination: Path) -> dict[str, Any]:
    """Write the local ignored artifact and return reproducibility metadata."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_version": MODEL_VERSION,
        "specification_sha256": specification_sha256(),
        "class_labels": list(CLASS_LABELS),
        "feature_columns": list(FROZEN_FEATURE_COLUMNS),
        "model": model,
    }
    joblib.dump(payload, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {
        "path": str(destination),
        "sha256": digest,
        "bytes": destination.stat().st_size,
        "specification_sha256": payload["specification_sha256"],
    }
