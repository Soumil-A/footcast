"""Consistent multiclass metrics for match-outcome models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

CLASS_LABELS = ("home_win", "draw", "away_win")


def validate_probabilities(
    probabilities: np.ndarray,
    *,
    row_count: int,
    class_count: int = len(CLASS_LABELS),
) -> np.ndarray:
    """Return a validated probability matrix."""
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (row_count, class_count):
        raise ValueError(
            "Probabilities must have shape "
            f"({row_count}, {class_count}); received {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Probabilities must contain only finite values")
    if (values < 0).any() or (values > 1).any():
        raise ValueError("Probabilities must be between zero and one")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("Every probability row must sum to one")
    return values


def evaluate_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    probabilities: np.ndarray,
    *,
    labels: tuple[str, ...] = CLASS_LABELS,
) -> dict[str, Any]:
    """Calculate the checkpoint metrics with a fixed class order."""
    actual = np.asarray(y_true, dtype=object)
    predicted = np.asarray(y_pred, dtype=object)
    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Targets and predictions must be one-dimensional")
    if len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError("Targets and predictions must have equal nonzero length")
    unknown = (set(actual) | set(predicted)) - set(labels)
    if unknown:
        raise ValueError(f"Unknown outcome labels: {sorted(unknown)}")

    probability_values = validate_probabilities(
        probabilities,
        row_count=len(actual),
        class_count=len(labels),
    )
    recalls = recall_score(
        actual,
        predicted,
        labels=labels,
        average=None,
        zero_division=0,
    )
    matrix = confusion_matrix(actual, predicted, labels=labels)
    label_positions = {label: index for index, label in enumerate(labels)}
    true_probabilities = probability_values[
        np.arange(len(actual)),
        [label_positions[label] for label in actual],
    ]
    clipped = np.clip(
        true_probabilities,
        np.finfo(float).eps,
        1.0,
    )
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "log_loss": float(-np.log(clipped).mean()),
        "per_class_recall": {
            label: float(value)
            for label, value in zip(labels, recalls, strict=True)
        },
        "confusion_matrix": matrix.astype(int).tolist(),
    }
