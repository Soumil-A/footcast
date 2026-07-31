"""Select and evaluate the Phase 4 checkpoint-two Random Forest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from footcast.data.download import DEFAULT_RAW_DIR
from footcast.data.manifest import DEFAULT_MANIFEST
from footcast.data.matches import load_match_statistics
from footcast.evaluation.metrics import evaluate_predictions
from footcast.evaluation.plots import (
    plot_feature_importances,
    plot_model_comparison,
    plot_single_confusion_matrix,
)
from footcast.features.build_features import (
    DEVELOPMENT_SPLITS,
    build_pre_match_features,
)
from footcast.models.random_forest import (
    N_ESTIMATORS,
    aligned_probabilities,
    evaluate_parameter_grid,
    feature_importances,
    make_random_forest,
    select_candidate,
)
from footcast.models.run_baselines import (
    MODEL_NAMES,
    compare_baselines,
    model_feature_columns,
    split_development_features,
)

PROJECT_ROOT = DEFAULT_MANIFEST.parents[1]
DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports" / "random_forest_results.json"
DEFAULT_REPORT_MARKDOWN = (
    PROJECT_ROOT / "reports" / "random_forest_results.md"
)
DEFAULT_CONFUSION_FIGURE = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "phase4"
    / "random_forest_confusion_matrix.png"
)
DEFAULT_COMPARISON_FIGURE = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "phase4"
    / "random_forest_model_comparison.png"
)
DEFAULT_IMPORTANCE_FIGURE = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "phase4"
    / "random_forest_feature_importance.png"
)
RANDOM_FOREST_NAME = "Random Forest"


def run_random_forest_experiment(features) -> dict[str, Any]:
    """Select within training, then compare once on chronological validation."""
    training, validation = split_development_features(features)
    columns = model_feature_columns(features)
    candidates = evaluate_parameter_grid(training, columns)
    selected = select_candidate(candidates)

    model = make_random_forest(selected["parameters"])
    model.fit(training[columns], training["result"])
    predictions = model.predict(validation[columns])
    probabilities = aligned_probabilities(model, validation[columns])
    forest_metrics = evaluate_predictions(
        validation["result"],
        predictions,
        probabilities,
    )

    baseline_report = compare_baselines(features)
    comparison = dict(baseline_report["results"])
    comparison[RANDOM_FOREST_NAME] = forest_metrics

    return {
        "status": "passed",
        "checkpoint": "Phase 4 checkpoint 2",
        "selection_metric": "mean_multiclass_log_loss",
        "selection_tie_breaker": "mean_macro_f1",
        "cross_validation": {
            "strategy": "expanding seasons with next-season validation",
            "fold_count": len(selected["folds"]),
            "minimum_training_seasons": 3,
            "candidate_count": len(candidates),
            "n_estimators": N_ESTIMATORS,
            "fixed_max_features": "sqrt",
            "candidates": candidates,
        },
        "selected_parameters": selected["parameters"],
        "selected_cross_validation": {
            "mean_log_loss": selected["mean_log_loss"],
            "mean_macro_f1": selected["mean_macro_f1"],
            "folds": selected["folds"],
        },
        "train_seasons": baseline_report["train_seasons"],
        "validation_seasons": baseline_report["validation_seasons"],
        "test_seasons_used": [],
        "holdout_seasons_used": [],
        "training_rows": int(len(training)),
        "validation_rows": int(len(validation)),
        "feature_count": len(columns),
        "results": comparison,
        "feature_importances": feature_importances(model, columns),
    }


def _parameter_text(parameters: dict[str, Any]) -> str:
    weight = parameters["class_weight"] or "none"
    depth = parameters["max_depth"]
    return (
        f"depth={depth}, leaf={parameters['min_samples_leaf']}, "
        f"class weight={weight}"
    )


def render_random_forest_report(report: dict[str, Any]) -> str:
    """Render selection evidence, final comparison, and limitations."""
    comparison_rows = []
    recall_rows = []
    model_names = (*MODEL_NAMES, RANDOM_FOREST_NAME)
    for name in model_names:
        metrics = report["results"][name]
        comparison_rows.append(
            f"| {name} | {metrics['accuracy']:.3f} | "
            f"{metrics['macro_f1']:.3f} | {metrics['log_loss']:.3f} |"
        )
        recall = metrics["per_class_recall"]
        recall_rows.append(
            f"| {name} | {recall['home_win']:.3f} | "
            f"{recall['draw']:.3f} | {recall['away_win']:.3f} |"
        )

    candidate_rows = []
    selected_parameters = report["selected_parameters"]
    for candidate in report["cross_validation"]["candidates"]:
        marker = (
            " **selected**"
            if candidate["parameters"] == selected_parameters
            else ""
        )
        candidate_rows.append(
            f"| {_parameter_text(candidate['parameters'])}{marker} | "
            f"{candidate['mean_log_loss']:.3f} | "
            f"{candidate['mean_macro_f1']:.3f} |"
        )

    fold_rows = []
    for fold in report["selected_cross_validation"]["folds"]:
        metrics = fold["metrics"]
        fold_rows.append(
            f"| {', '.join(fold['train_seasons'])} | "
            f"{fold['validation_season']} | {metrics['accuracy']:.3f} | "
            f"{metrics['macro_f1']:.3f} | {metrics['log_loss']:.3f} |"
        )

    importance_rows = [
        f"| {item['feature']} | {item['importance']:.4f} |"
        for item in report["feature_importances"][:15]
    ]
    parameters = _parameter_text(report["selected_parameters"])
    return f"""# FootCast Phase 4 Random Forest Report

**Status:** {str(report["status"]).upper()}

Checkpoint 2 selects Random Forest settings using expanding chronological folds
inside the training period, refits the selected configuration on every training
row, and compares it once on 2023-24 validation. The 2024-25 test and 2025-26
holdout seasons were neither loaded nor inspected.

- Training rows: {report["training_rows"]}
- Validation rows: {report["validation_rows"]}
- Numeric pre-match features: {report["feature_count"]}
- Candidate configurations: {report["cross_validation"]["candidate_count"]}
- Expanding folds per candidate: {report["cross_validation"]["fold_count"]}
- Trees per forest: {report["cross_validation"]["n_estimators"]}
- Selection rule: lowest mean log loss, then highest macro F1
- Selected configuration: {parameters}

## Training-period model selection

| Candidate | Mean log loss | Mean macro F1 |
| --- | ---: | ---: |
{chr(10).join(candidate_rows)}

The folds always train on earlier seasons and validate on the immediately
following season. No fold trains on data later than its validation season.

| Training seasons | Validation season | Accuracy | Macro F1 | Log loss |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(fold_rows)}

## 2023-24 validation comparison

| Model | Accuracy | Macro F1 | Log loss |
| --- | ---: | ---: | ---: |
{chr(10).join(comparison_rows)}

## Per-class recall

| Model | Home win | Draw | Away win |
| --- | ---: | ---: | ---: |
{chr(10).join(recall_rows)}

## Largest training-derived feature importances

| Feature | Impurity importance |
| --- | ---: |
{chr(10).join(importance_rows)}

Impurity importance describes how often a fitted forest uses a feature to
reduce node impurity. Correlated features can divide or distort importance, so
this is a diagnostic rather than a causal explanation.

The model-comparison chart keeps label metrics separate from log loss because
higher is better for accuracy and macro F1 while lower is better for log loss.
The confusion matrix shows the selected forest's class errors. Probability
calibration, detailed subgroup analysis, and reserved-season evaluation remain
outside this checkpoint.
"""


def run_random_forest_pipeline(
    raw_dir: Path = DEFAULT_RAW_DIR,
    report_json: Path = DEFAULT_REPORT_JSON,
    report_markdown: Path = DEFAULT_REPORT_MARKDOWN,
    confusion_figure: Path = DEFAULT_CONFUSION_FIGURE,
    comparison_figure: Path = DEFAULT_COMPARISON_FIGURE,
    importance_figure: Path = DEFAULT_IMPORTANCE_FIGURE,
) -> dict[str, Any]:
    """Regenerate development features, select the forest, and write evidence."""
    matches = load_match_statistics(raw_dir, splits=DEVELOPMENT_SPLITS)
    features = build_pre_match_features(matches)
    report = run_random_forest_experiment(features)

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_markdown.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    report_markdown.write_text(
        render_random_forest_report(report),
        encoding="utf-8",
    )
    plot_single_confusion_matrix(
        report["results"][RANDOM_FOREST_NAME],
        confusion_figure,
    )
    plot_model_comparison(report["results"], comparison_figure)
    plot_feature_importances(
        report["feature_importances"],
        importance_figure,
    )
    return report


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report = run_random_forest_pipeline()
    metrics = report["results"][RANDOM_FOREST_NAME]
    print(
        "Selected Random Forest with "
        f"validation macro F1={metrics['macro_f1']:.3f} and "
        f"log loss={metrics['log_loss']:.3f}."
    )


if __name__ == "__main__":
    main()
