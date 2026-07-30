"""Plots for model-comparison reports."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/footcast-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from footcast.evaluation.metrics import CLASS_LABELS


def plot_confusion_matrices(
    results: dict[str, dict[str, Any]],
    destination: Path,
) -> None:
    """Write consistently ordered confusion matrices for all baselines."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    display_labels = ["Home win", "Draw", "Away win"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

    for axis, (model_name, metrics) in zip(
        axes.flat, results.items(), strict=True
    ):
        sns.heatmap(
            metrics["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=display_labels,
            yticklabels=display_labels,
            ax=axis,
        )
        axis.set_title(model_name)
        axis.set_xlabel("Predicted outcome")
        axis.set_ylabel("Actual outcome")

    figure.suptitle(
        "FootCast baseline confusion matrices — 2023-24 validation",
        fontsize=14,
    )
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)

    if len(CLASS_LABELS) != 3:
        raise AssertionError("Plot layout assumes three outcome classes")
