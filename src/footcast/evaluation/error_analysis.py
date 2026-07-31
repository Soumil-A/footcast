"""Validation-only error slices for FootCast probability models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from footcast.evaluation.metrics import CLASS_LABELS, evaluate_predictions


def _slice_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    metrics = evaluate_predictions(
        actual[mask],
        predicted[mask],
        probabilities[mask],
    )
    return {
        "rows": int(mask.sum()),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "log_loss": metrics["log_loss"],
        "multiclass_brier_score": metrics["multiclass_brier_score"],
        "expected_calibration_error": metrics[
            "expected_calibration_error"
        ],
        "mean_confidence": float(probabilities[mask].max(axis=1).mean()),
    }


def _group_metrics(
    values: pd.Series,
    actual: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, dict[str, Any]]:
    groups = {}
    for value in values.dropna().unique():
        mask = values.eq(value).to_numpy()
        groups[str(value)] = _slice_metrics(
            actual,
            predicted,
            probabilities,
            mask,
        )
    return groups


def build_error_analysis(
    validation: pd.DataFrame,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """Analyze model errors without using validation outcomes for fitting."""
    required = {
        "season",
        "match_date",
        "home_team",
        "away_team",
        "result",
        "home_history_matches",
        "away_history_matches",
        "home_season_matches_played",
        "away_season_matches_played",
        "elo_difference",
    }
    missing = sorted(required - set(validation.columns))
    if missing:
        raise ValueError(f"Error analysis is missing columns: {missing}")
    if len(validation) != len(probabilities):
        raise ValueError("Validation rows and probabilities must align")

    frame = validation.reset_index(drop=True).copy()
    actual = frame["result"].to_numpy(dtype=object)
    predicted = np.asarray(CLASS_LABELS, dtype=object)[
        np.argmax(probabilities, axis=1)
    ]
    confidence = probabilities.max(axis=1)
    frame["predicted"] = predicted
    frame["confidence"] = confidence
    frame["correct"] = actual == predicted
    frame["history_slice"] = np.where(
        (frame["home_history_matches"] == 0)
        | (frame["away_history_matches"] == 0),
        "cold_start",
        "known_history",
    )
    frame["season_timing"] = np.where(
        frame[
            ["home_season_matches_played", "away_season_matches_played"]
        ].min(axis=1)
        < 5,
        "early_season",
        "established_season",
    )
    frame["elo_gap"] = pd.cut(
        frame["elo_difference"].abs(),
        bins=[-np.inf, 50, 100, np.inf],
        labels=["close_0_50", "medium_50_100", "large_over_100"],
    ).astype("object")
    frame["confidence_band"] = pd.cut(
        confidence,
        bins=[0.0, 0.45, 0.60, 1.0],
        labels=["low_up_to_45pct", "medium_45_60pct", "high_over_60pct"],
        include_lowest=True,
    ).astype("object")

    high_confidence_mistakes = frame.loc[
        (~frame["correct"]) & (frame["confidence"] >= 0.60)
    ].sort_values("confidence", ascending=False)
    mistake_rows = []
    for row in high_confidence_mistakes.head(15).itertuples():
        index = int(row.Index)
        mistake_rows.append(
            {
                "season": str(row.season),
                "match_date": str(row.match_date),
                "home_team": str(row.home_team),
                "away_team": str(row.away_team),
                "actual": str(row.result),
                "predicted": str(row.predicted),
                "confidence": float(row.confidence),
                "probabilities": {
                    label: float(probabilities[index, class_index])
                    for class_index, label in enumerate(CLASS_LABELS)
                },
            }
        )

    return {
        "overall": _slice_metrics(
            actual,
            predicted,
            probabilities,
            np.ones(len(frame), dtype=bool),
        ),
        "by_actual_outcome": _group_metrics(
            frame["result"],
            actual,
            predicted,
            probabilities,
        ),
        "by_history": _group_metrics(
            frame["history_slice"],
            actual,
            predicted,
            probabilities,
        ),
        "by_season_timing": _group_metrics(
            frame["season_timing"],
            actual,
            predicted,
            probabilities,
        ),
        "by_elo_gap": _group_metrics(
            frame["elo_gap"],
            actual,
            predicted,
            probabilities,
        ),
        "by_confidence": _group_metrics(
            frame["confidence_band"],
            actual,
            predicted,
            probabilities,
        ),
        "high_confidence_mistake_count": int(
            len(high_confidence_mistakes)
        ),
        "highest_confidence_mistakes": mistake_rows,
    }
