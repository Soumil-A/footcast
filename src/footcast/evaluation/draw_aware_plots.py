"""Figures for the Phase 6 draw-aware decision checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def plot_draw_aware_comparison(
    models: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot probability and draw metrics for the decision candidates."""
    names = list(models)
    log_losses = [models[name]["mean_log_loss"] for name in names]
    brier_scores = [
        models[name]["mean_multiclass_brier_score"] for name in names
    ]
    macro_f1 = [models[name]["mean_macro_f1"] for name in names]
    draw_recall = [models[name]["mean_draw_recall"] for name in names]

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].bar(names, log_losses, label="Log loss")
    axes[0].scatter(names, brier_scores, color="black", label="Brier score")
    axes[0].set_ylim(0.5, 1.1)
    axes[0].set_title("Probability metrics (lower is better)")
    axes[0].legend()
    axes[1].bar(names, macro_f1, label="Macro F1")
    axes[1].scatter(names, draw_recall, color="black", label="Draw recall")
    axes[1].set_ylim(0.0, 0.55)
    axes[1].set_title("Label metrics (higher is better)")
    axes[1].legend()
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("FootCast Phase 6 draw-aware checkpoint")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_draw_aware_by_season(
    models: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot next-season log loss for each decision candidate."""
    figure, axis = plt.subplots(figsize=(12, 6))
    for name, values in models.items():
        folds = values["folds"]
        axis.plot(
            [fold["evaluation_season"] for fold in folds],
            [fold["metrics"]["log_loss"] for fold in folds],
            marker="o",
            label=name,
        )
    axis.set_title("Draw-aware next-season log loss")
    axis.set_xlabel("Evaluation season")
    axis.set_ylabel("Multiclass log loss (lower is better)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_draw_weight_tradeoff(
    candidates: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Show the probability-quality cost of increasing draw recall."""
    figure, axis = plt.subplots(figsize=(8, 6))
    for candidate in candidates:
        weight = candidate["parameters"]["draw_weight"]
        axis.scatter(
            candidate["mean_draw_recall"],
            candidate["mean_log_loss"],
            s=90,
        )
        axis.annotate(
            f"weight {weight:.2f}",
            (
                candidate["mean_draw_recall"],
                candidate["mean_log_loss"],
            ),
            xytext=(6, 6),
            textcoords="offset points",
        )
    axis.axhline(0.970, color="tab:red", linestyle="--", label="Log-loss gate")
    axis.axvline(0.100, color="tab:green", linestyle="--", label="Draw gate")
    axis.set_title("Draw weighting trades probability quality for recall")
    axis.set_xlabel("Mean draw recall (higher is better)")
    axis.set_ylabel("Mean log loss (lower is better)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
