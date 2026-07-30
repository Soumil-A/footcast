"""Train and compare Phase 4 checkpoint-one baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from footcast.config import DATA_SPLIT
from footcast.data.download import DEFAULT_RAW_DIR
from footcast.data.manifest import DEFAULT_MANIFEST
from footcast.data.matches import load_match_statistics
from footcast.evaluation.metrics import CLASS_LABELS, evaluate_predictions
from footcast.evaluation.plots import plot_confusion_matrices
from footcast.features.build_features import (
    DEVELOPMENT_SPLITS,
    IDENTIFIER_COLUMNS,
    build_pre_match_features,
)
from footcast.models.baselines import (
    AlwaysHomeBaseline,
    EloBaseline,
    LogisticRegressionBaseline,
    MajorityClassBaseline,
)

PROJECT_ROOT = DEFAULT_MANIFEST.parents[1]
DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports" / "baseline_results.json"
DEFAULT_REPORT_MARKDOWN = PROJECT_ROOT / "reports" / "baseline_results.md"
DEFAULT_FIGURE = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "phase4"
    / "baseline_confusion_matrices.png"
)
MODEL_NAMES = (
    "Majority class",
    "Always home",
    "Elo",
    "Logistic regression",
)


def split_development_features(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enforce the frozen chronological train/validation boundary."""
    required = {*IDENTIFIER_COLUMNS, "result"}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Feature table is missing columns: {missing}")

    observed_splits = set(features["split"])
    forbidden = observed_splits & {"test", "holdout"}
    if forbidden:
        raise ValueError(
            f"Test or holdout data entered baseline modeling: {sorted(forbidden)}"
        )
    unexpected = observed_splits - DEVELOPMENT_SPLITS
    if unexpected:
        raise ValueError(f"Unexpected data splits: {sorted(unexpected)}")

    training = features.loc[features["split"] == "train"].copy()
    validation = features.loc[features["split"] == "validation"].copy()
    if training.empty or validation.empty:
        raise ValueError("Both training and validation rows are required")
    if set(training["season"]) != set(DATA_SPLIT.train):
        raise ValueError("Training seasons do not match the frozen split contract")
    if set(validation["season"]) != set(DATA_SPLIT.validation):
        raise ValueError(
            "Validation seasons do not match the frozen split contract"
        )
    if training["match_date"].max() >= validation["match_date"].min():
        raise ValueError("Training must end before validation begins")
    return training, validation


def model_feature_columns(features: pd.DataFrame) -> list[str]:
    """Select the numeric pre-match predictors, excluding IDs and target."""
    columns = [
        column
        for column in features.columns
        if column not in {*IDENTIFIER_COLUMNS, "result"}
    ]
    nonnumeric = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(features[column])
    ]
    if nonnumeric:
        raise ValueError(f"Model features must be numeric: {nonnumeric}")
    if not columns:
        raise ValueError("No model feature columns were found")
    return columns


def compare_baselines(features: pd.DataFrame) -> dict[str, Any]:
    """Fit on training seasons and score once on the validation season."""
    training, validation = split_development_features(features)
    feature_columns = model_feature_columns(features)
    x_train = training[feature_columns]
    y_train = training["result"]
    x_validation = validation[feature_columns]
    y_validation = validation["result"]

    models = {
        "Majority class": MajorityClassBaseline(),
        "Always home": AlwaysHomeBaseline(),
        "Elo": EloBaseline(),
        "Logistic regression": LogisticRegressionBaseline(),
    }
    results: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_validation)
        probabilities = model.predict_proba(x_validation)
        results[name] = evaluate_predictions(
            y_validation,
            predictions,
            probabilities,
        )

    return {
        "status": "passed",
        "checkpoint": "Phase 4 checkpoint 1",
        "train_seasons": list(DATA_SPLIT.train),
        "validation_seasons": list(DATA_SPLIT.validation),
        "test_seasons_used": [],
        "holdout_seasons_used": [],
        "training_rows": int(len(training)),
        "validation_rows": int(len(validation)),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "class_order": list(CLASS_LABELS),
        "training_class_distribution": {
            label: int((y_train == label).sum()) for label in CLASS_LABELS
        },
        "validation_class_distribution": {
            label: int((y_validation == label).sum()) for label in CLASS_LABELS
        },
        "results": results,
    }


def render_baseline_report(report: dict[str, Any]) -> str:
    """Render the model comparison and its interpretation."""
    rows = []
    recall_rows = []
    for model_name in MODEL_NAMES:
        metrics = report["results"][model_name]
        rows.append(
            f"| {model_name} | {metrics['accuracy']:.3f} | "
            f"{metrics['macro_f1']:.3f} | {metrics['log_loss']:.3f} |"
        )
        recall = metrics["per_class_recall"]
        recall_rows.append(
            f"| {model_name} | {recall['home_win']:.3f} | "
            f"{recall['draw']:.3f} | {recall['away_win']:.3f} |"
        )

    return f"""# FootCast Phase 4 Baseline Report

**Status:** {str(report["status"]).upper()}

This first modeling checkpoint compares four deliberately simple methods on the
same chronological boundary. Models fit on 2015-16 through 2022-23 and are
evaluated on 2023-24. The 2024-25 test and 2025-26 holdout seasons were neither
loaded nor inspected.

- Training rows: {report["training_rows"]}
- Validation rows: {report["validation_rows"]}
- Leakage-safe numeric features: {report["feature_count"]}
- Class order: {", ".join(report["class_order"])}

## Overall validation metrics

| Model | Accuracy | Macro F1 | Log loss |
| --- | ---: | ---: | ---: |
{chr(10).join(rows)}

## Per-class recall

| Model | Home win | Draw | Away win |
| --- | ---: | ---: | ---: |
{chr(10).join(recall_rows)}

Accuracy is not sufficient on its own: a model can score reasonably by mostly
choosing the common home-win class while failing to recognize draws or away
wins. Macro F1 gives each outcome equal influence, per-class recall exposes
which outcomes are missed, and log loss evaluates the quality and confidence
of all three probabilities. Lower log loss is better; higher accuracy, macro
F1, and recall are better.

The majority baseline uses training outcome frequencies as probabilities. The
always-home baseline is intentionally deterministic, so confident mistakes
receive a severe log-loss penalty. Elo uses only the two pre-match ratings and
the training draw rate. Logistic regression uses all leakage-safe numeric
features; its median imputer and scaler are fitted on training rows only.

See `reports/figures/phase4/baseline_confusion_matrices.png` for counts by
actual and predicted class. These validation results are a development
checkpoint, not a final estimate of future performance.
"""


def run_baseline_pipeline(
    raw_dir: Path = DEFAULT_RAW_DIR,
    report_json: Path = DEFAULT_REPORT_JSON,
    report_markdown: Path = DEFAULT_REPORT_MARKDOWN,
    figure_path: Path = DEFAULT_FIGURE,
) -> dict[str, Any]:
    """Regenerate features in memory, compare models, and write reports."""
    matches = load_match_statistics(raw_dir, splits=DEVELOPMENT_SPLITS)
    features = build_pre_match_features(matches)
    report = compare_baselines(features)

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_markdown.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    report_markdown.write_text(
        render_baseline_report(report),
        encoding="utf-8",
    )
    plot_confusion_matrices(report["results"], figure_path)
    return report


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report = run_baseline_pipeline()
    best = max(
        report["results"],
        key=lambda name: report["results"][name]["macro_f1"],
    )
    print(
        f"Compared {len(report['results'])} baselines on "
        f"{report['validation_rows']} validation matches. "
        f"Best macro F1: {best}."
    )


if __name__ == "__main__":
    main()
