"""Reliability and validation error-analysis figures."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/footcast-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from footcast.evaluation.metrics import CLASS_LABELS


def plot_reliability_comparison(
    uncalibrated: dict[str, list[dict[str, float | int]]],
    calibrated: dict[str, list[dict[str, float | int]]],
    destination: Path,
    *,
    selected_method: str = "selected method",
    evaluation_label: str = "2023-24 validation",
) -> None:
    """Plot classwise observed frequency before and after calibration."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    display_names = {
        "home_win": "Home win",
        "draw": "Draw",
        "away_win": "Away win",
    }
    rows = (
        (("Selected: uncalibrated", uncalibrated),)
        if selected_method.startswith("uncalibrated")
        else (
            ("Uncalibrated Random Forest", uncalibrated),
            (f"Selected: {selected_method}", calibrated),
        )
    )
    figure, axes = plt.subplots(
        len(rows),
        3,
        figsize=(13, 4 * len(rows)),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, (title, values) in enumerate(rows):
        for column_index, label in enumerate(CLASS_LABELS):
            axis = axes[row_index, column_index]
            bins = values[label]
            axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
            axis.plot(
                [float(item["mean_probability"]) for item in bins],
                [float(item["observed_frequency"]) for item in bins],
                marker="o",
                linewidth=2,
            )
            for item in bins:
                axis.annotate(
                    str(item["count"]),
                    (
                        float(item["mean_probability"]),
                        float(item["observed_frequency"]),
                    ),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                )
            axis.set_title(f"{title}\n{display_names[label]}")
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.grid(alpha=0.2)
            if row_index == len(rows) - 1:
                axis.set_xlabel("Mean predicted probability")
            if column_index == 0:
                axis.set_ylabel("Observed frequency")
    figure.suptitle(
        f"FootCast classwise reliability — {evaluation_label}",
        fontsize=14,
    )
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_error_slices(
    analysis: dict[str, Any],
    destination: Path,
    *,
    title: str = "Selected probability model error slices — 2023-24 validation",
) -> None:
    """Compare accuracy and confidence for the principal validation slices."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    panels = (
        ("Actual outcome", analysis["by_actual_outcome"]),
        ("Season timing", analysis["by_season_timing"]),
        ("Absolute Elo gap", analysis["by_elo_gap"]),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    width = 0.36
    for axis, (panel_title, groups) in zip(axes, panels, strict=True):
        names = list(groups)
        positions = np.arange(len(names))
        axis.bar(
            positions - width / 2,
            [groups[name]["accuracy"] for name in names],
            width,
            label="Accuracy",
        )
        axis.bar(
            positions + width / 2,
            [groups[name]["mean_confidence"] for name in names],
            width,
            label="Mean confidence",
        )
        axis.set_title(panel_title)
        axis.set_ylim(0, 1)
        axis.set_xticks(positions, names, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Rate")
    axes[-1].legend()
    figure.suptitle(title, fontsize=14)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
