"""Plots for model-comparison reports."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/footcast-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
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


def plot_single_confusion_matrix(
    metrics: dict[str, Any],
    destination: Path,
) -> None:
    """Write one fixed-order confusion matrix for the selected forest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    display_labels = ["Home win", "Draw", "Away win"]
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    sns.heatmap(
        metrics["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar=False,
        xticklabels=display_labels,
        yticklabels=display_labels,
        ax=axis,
    )
    axis.set_title("Selected Random Forest — 2023-24 validation")
    axis.set_xlabel("Predicted outcome")
    axis.set_ylabel("Actual outcome")
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_model_comparison(
    results: dict[str, dict[str, Any]],
    destination: Path,
) -> None:
    """Compare label quality and probability quality without mixing scales."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = list(results)
    positions = np.arange(len(names))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    axes[0].bar(
        positions - width / 2,
        [results[name]["accuracy"] for name in names],
        width,
        label="Accuracy",
    )
    axes[0].bar(
        positions + width / 2,
        [results[name]["macro_f1"] for name in names],
        width,
        label="Macro F1",
    )
    axes[0].set_ylim(0, 0.7)
    axes[0].set_title("Outcome-label metrics (higher is better)")
    axes[0].set_xticks(positions, names, rotation=25, ha="right")
    axes[0].legend()

    log_losses = [results[name]["log_loss"] for name in names]
    competitive_losses = [
        value
        for name, value in zip(names, log_losses, strict=True)
        if name != "Always home"
    ]
    display_ceiling = max(max(competitive_losses) * 1.12, 0.1)
    bars = axes[1].bar(
        positions,
        [min(value, display_ceiling) for value in log_losses],
        color=sns.color_palette("muted", len(names)),
    )
    for bar, value in zip(bars, log_losses, strict=True):
        label = f"{value:.3f}"
        if value > display_ceiling:
            label += "\n(off scale)"
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            min(value, display_ceiling) + display_ceiling * 0.015,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1].set_ylim(0, display_ceiling * 1.13)
    axes[1].set_title("Multiclass log loss (lower is better)")
    axes[1].set_xticks(positions, names, rotation=25, ha="right")
    figure.suptitle("FootCast model comparison — 2023-24 validation")
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_feature_importances(
    importances: list[dict[str, Any]],
    destination: Path,
    *,
    limit: int = 15,
) -> None:
    """Plot the selected forest's largest training-derived importances."""
    if not importances:
        raise ValueError("At least one feature importance is required")
    selected = list(reversed(importances[:limit]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    axis.barh(
        [item["feature"] for item in selected],
        [item["importance"] for item in selected],
        color=sns.color_palette("crest", len(selected)),
    )
    axis.set_title("Selected Random Forest feature importance")
    axis.set_xlabel("Mean decrease in impurity")
    axis.set_ylabel("Training-derived feature")
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
