"""Tests for the frozen v1 contract and final-evaluation boundary."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from footcast.config import DATA_SPLIT
from footcast.models.frozen import (
    FROZEN_FEATURE_COLUMNS,
    MODEL_VERSION,
    fit_frozen_model,
    frozen_specification,
    save_frozen_artifact,
    specification_sha256,
)
from footcast.models.run_final_test import (
    split_final_evaluation_features,
    verify_frozen_specification,
)


def _final_evaluation_frame() -> pd.DataFrame:
    rows = []
    seasons = DATA_SPLIT.train + DATA_SPLIT.validation + DATA_SPLIT.test
    labels = ("home_win", "draw", "away_win")
    for season_index, season in enumerate(seasons):
        split = (
            "train"
            if season in DATA_SPLIT.train
            else "validation"
            if season in DATA_SPLIT.validation
            else "test"
        )
        year = 2015 + season_index
        for label_index, label in enumerate(labels):
            row = {
                "season": season,
                "split": split,
                "match_date": f"{year}-08-{label_index + 1:02d}",
                "home_team": f"Home {season_index} {label_index}",
                "away_team": f"Away {season_index} {label_index}",
                "result": label,
            }
            row.update(
                {
                    column: float(season_index + label_index)
                    for column in FROZEN_FEATURE_COLUMNS
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_tracked_specification_matches_executable_contract(tmp_path) -> None:
    path = tmp_path / "spec.json"
    path.write_text(
        json.dumps(frozen_specification(), indent=2),
        encoding="utf-8",
    )

    verify_frozen_specification(path)

    assert len(FROZEN_FEATURE_COLUMNS) == 38
    assert len(specification_sha256()) == 64


def test_specification_mismatch_stops_final_evaluation(tmp_path) -> None:
    path = tmp_path / "spec.json"
    changed = frozen_specification()
    changed["calibration"] = "isotonic"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="contract differ"):
        verify_frozen_specification(path)


def test_final_split_refits_on_development_and_tests_once() -> None:
    development, test = split_final_evaluation_features(
        _final_evaluation_frame()
    )

    assert set(development["season"]) == set(
        DATA_SPLIT.train + DATA_SPLIT.validation
    )
    assert set(test["season"]) == set(DATA_SPLIT.test)
    assert not set(development["split"]) & {"test", "holdout"}
    assert test["split"].unique().tolist() == ["test"]


def test_final_split_rejects_holdout_rows() -> None:
    frame = _final_evaluation_frame()
    leaked = frame.iloc[[0]].copy()
    leaked["season"] = DATA_SPLIT.holdout[0]
    leaked["split"] = "holdout"

    with pytest.raises(ValueError, match="Holdout"):
        split_final_evaluation_features(pd.concat([frame, leaked]))


def test_final_split_rejects_feature_drift() -> None:
    frame = _final_evaluation_frame().drop(columns=[FROZEN_FEATURE_COLUMNS[-1]])

    with pytest.raises(ValueError, match="missing columns"):
        split_final_evaluation_features(frame)


def test_frozen_artifact_round_trip(tmp_path) -> None:
    target = np.asarray(
        ["home_win", "draw", "away_win"] * 4,
        dtype=object,
    )
    features = pd.DataFrame(
        {
            column: np.arange(len(target), dtype=float)
            for column in FROZEN_FEATURE_COLUMNS
        }
    )
    features.loc[0, "rest_days_difference"] = np.nan
    model = fit_frozen_model(features, target)
    destination = tmp_path / "model.joblib"

    metadata = save_frozen_artifact(model, destination)
    payload = joblib.load(destination)

    assert payload["model_version"] == MODEL_VERSION
    assert payload["specification_sha256"] == specification_sha256()
    assert payload["feature_columns"] == list(FROZEN_FEATURE_COLUMNS)
    assert metadata["bytes"] > 0
    assert len(metadata["sha256"]) == 64
    assert payload["model"].predict_proba(features).shape == (12, 3)
