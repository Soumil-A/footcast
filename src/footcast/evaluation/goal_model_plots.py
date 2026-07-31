"""Figures for Phase 6 rolling goal-model comparisons."""

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


def plot_goal_model_comparison(
    models: dict[str, dict[str, Any]],
    destination: Path,
) -> None:
    """Compare mean rolling label and probability metrics."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = list(models)
    positions = np.arange(len(names))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].bar(
        positions - width / 2,
        [models[name]["mean_accuracy"] for name in names],
        width,
        label="Accuracy",
    )
    axes[0].bar(
        positions + width / 2,
        [models[name]["mean_macro_f1"] for name in names],
        width,
        label="Macro F1",
    )
    axes[0].set_ylim(0, 0.7)
    axes[0].set_title("Rolling label metrics (higher is better)")
    axes[0].set_xticks(positions, names, rotation=25, ha="right")
    axes[0].legend()

    bars = axes[1].bar(
        positions,
        [models[name]["mean_log_loss"] for name in names],
        color=sns.color_palette("muted", len(names)),
    )
    for bar, name in zip(bars, names, strict=True):
        value = models[name]["mean_log_loss"]
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1].set_ylim(0.8, 1.15)
    axes[1].set_title("Mean rolling log loss (lower is better)")
    axes[1].set_xticks(positions, names, rotation=25, ha="right")
    figure.suptitle("FootCast Phase 6 rolling model comparison")
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_log_loss_by_season(
    models: dict[str, dict[str, Any]],
    destination: Path,
) -> None:
    """Show whether model probability quality is stable across seasons."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for name, values in models.items():
        axis.plot(
            [fold["evaluation_season"] for fold in values["folds"]],
            [fold["metrics"]["log_loss"] for fold in values["folds"]],
            marker="o",
            label=name,
        )
    axis.set_title("Next-season log loss by rolling backtest")
    axis.set_xlabel("Evaluation season")
    axis.set_ylabel("Multiclass log loss (lower is better)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_draw_recall(
    models: dict[str, dict[str, Any]],
    destination: Path,
) -> None:
    """Compare each model's mean ability to select actual draws."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = list(models)
    values = [models[name]["mean_draw_recall"] for name in names]
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    bars = axis.bar(names, values, color=sns.color_palette("deep", len(names)))
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    axis.set_ylim(0, max(max(values) * 1.25, 0.1))
    axis.set_title("Mean draw recall across rolling backtests")
    axis.set_ylabel("Draw recall")
    axis.tick_params(axis="x", rotation=25)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
