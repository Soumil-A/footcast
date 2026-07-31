"""Consistent multiclass metrics for match-outcome models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
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
    precisions = precision_score(
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
    one_hot = np.zeros_like(probability_values)
    one_hot[
        np.arange(len(actual)),
        [label_positions[label] for label in actual],
    ] = 1.0
    confidence = probability_values.max(axis=1)
    correctness = (actual == predicted).astype(float)
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
        "multiclass_brier_score": float(
            np.square(probability_values - one_hot).sum(axis=1).mean()
        ),
        "expected_calibration_error": expected_calibration_error(
            correctness,
            confidence,
        ),
        "per_class_recall": {
            label: float(value)
            for label, value in zip(labels, recalls, strict=True)
        },
        "per_class_precision": {
            label: float(value)
            for label, value in zip(labels, precisions, strict=True)
        },
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def expected_calibration_error(
    correctness: np.ndarray,
    confidence: np.ndarray,
    *,
    bin_count: int = 10,
) -> float:
    """Return confidence-weighted calibration error across equal-width bins."""
    correct = np.asarray(correctness, dtype=float)
    predicted_confidence = np.asarray(confidence, dtype=float)
    if (
        correct.ndim != 1
        or predicted_confidence.ndim != 1
        or len(correct) == 0
        or len(correct) != len(predicted_confidence)
    ):
        raise ValueError("Correctness and confidence require equal nonzero vectors")
    if not 1 <= bin_count <= 100:
        raise ValueError("bin_count must be between 1 and 100")
    if (predicted_confidence < 0).any() or (predicted_confidence > 1).any():
        raise ValueError("Confidence must be between zero and one")

    edges = np.linspace(0.0, 1.0, bin_count + 1)
    assignments = np.minimum(
        np.digitize(predicted_confidence, edges[1:-1], right=True),
        bin_count - 1,
    )
    error = 0.0
    for bin_index in range(bin_count):
        mask = assignments == bin_index
        if not mask.any():
            continue
        error += float(mask.mean()) * abs(
            float(correct[mask].mean())
            - float(predicted_confidence[mask].mean())
        )
    return float(error)


def classwise_calibration_bins(
    y_true: Sequence[str],
    probabilities: np.ndarray,
    *,
    labels: tuple[str, ...] = CLASS_LABELS,
    bin_count: int = 10,
) -> dict[str, list[dict[str, float | int]]]:
    """Summarize predicted versus observed frequency for each outcome."""
    actual = np.asarray(y_true, dtype=object)
    if len(actual) == 0:
        raise ValueError("Calibration bins require at least one target")
    if not 1 <= bin_count <= 100:
        raise ValueError("bin_count must be between 1 and 100")
    values = validate_probabilities(
        probabilities,
        row_count=len(actual),
        class_count=len(labels),
    )
    unknown = set(actual) - set(labels)
    if unknown:
        raise ValueError(f"Unknown outcome labels: {sorted(unknown)}")
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    result: dict[str, list[dict[str, float | int]]] = {}
    for class_index, label in enumerate(labels):
        predicted = values[:, class_index]
        observed = (actual == label).astype(float)
        assignments = np.minimum(
            np.digitize(predicted, edges[1:-1], right=True),
            bin_count - 1,
        )
        bins = []
        for bin_index in range(bin_count):
            mask = assignments == bin_index
            if not mask.any():
                continue
            bins.append(
                {
                    "lower": float(edges[bin_index]),
                    "upper": float(edges[bin_index + 1]),
                    "count": int(mask.sum()),
                    "mean_probability": float(predicted[mask].mean()),
                    "observed_frequency": float(observed[mask].mean()),
                }
            )
        result[label] = bins
    return result
