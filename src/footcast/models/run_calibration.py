"""Run Phase 5 validation-only calibration and error analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from footcast.data.download import DEFAULT_RAW_DIR
from footcast.data.manifest import DEFAULT_MANIFEST
from footcast.data.matches import load_match_statistics
from footcast.evaluation.calibration_plots import (
    plot_error_slices,
    plot_reliability_comparison,
)
from footcast.evaluation.error_analysis import build_error_analysis
from footcast.evaluation.metrics import (
    CLASS_LABELS,
    classwise_calibration_bins,
    evaluate_predictions,
)
from footcast.features.build_features import (
    DEVELOPMENT_SPLITS,
    build_pre_match_features,
)
from footcast.models.calibration import (
    evaluate_calibration_methods,
    generate_oof_predictions,
    make_calibrator,
    select_calibration_method,
)
from footcast.models.random_forest import (
    SELECTED_PARAMETERS,
    aligned_probabilities,
    make_random_forest,
)
from footcast.models.run_baselines import (
    model_feature_columns,
    split_development_features,
)

PROJECT_ROOT = DEFAULT_MANIFEST.parents[1]
DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports" / "calibration_results.json"
DEFAULT_REPORT_MARKDOWN = (
    PROJECT_ROOT / "reports" / "calibration_results.md"
)
DEFAULT_RELIABILITY_FIGURE = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "phase5"
    / "reliability_comparison.png"
)
DEFAULT_ERROR_FIGURE = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "phase5"
    / "validation_error_slices.png"
)


def _predictions(probabilities: np.ndarray) -> np.ndarray:
    return np.asarray(CLASS_LABELS, dtype=object)[
        np.argmax(probabilities, axis=1)
    ]


def run_calibration_experiment(features) -> dict[str, Any]:
    """Select calibration inside training and analyze 2023-24 validation."""
    training, validation = split_development_features(features)
    columns = model_feature_columns(features)
    oof_folds = generate_oof_predictions(training, columns)
    candidates = evaluate_calibration_methods(oof_folds)
    selected = select_calibration_method(candidates)

    calibration_probabilities = np.vstack(
        [fold.probabilities for fold in oof_folds]
    )
    calibration_target = np.concatenate(
        [fold.target for fold in oof_folds]
    )
    calibrator = make_calibrator(str(selected["method"])).fit(
        calibration_probabilities,
        calibration_target,
    )

    forest = make_random_forest(SELECTED_PARAMETERS)
    forest.fit(training[columns], training["result"])
    uncalibrated_probabilities = aligned_probabilities(
        forest,
        validation[columns],
    )
    calibrated_probabilities = calibrator.transform(
        uncalibrated_probabilities
    )
    actual = validation["result"].to_numpy(dtype=object)
    uncalibrated_metrics = evaluate_predictions(
        actual,
        _predictions(uncalibrated_probabilities),
        uncalibrated_probabilities,
    )
    calibrated_metrics = evaluate_predictions(
        actual,
        _predictions(calibrated_probabilities),
        calibrated_probabilities,
    )
    calibration_bins = {
        "uncalibrated": classwise_calibration_bins(
            actual,
            uncalibrated_probabilities,
        ),
        "selected": classwise_calibration_bins(
            actual,
            calibrated_probabilities,
        ),
    }
    analysis = build_error_analysis(
        validation,
        calibrated_probabilities,
    )

    return {
        "status": "passed",
        "checkpoint": "Phase 5 checkpoint 1",
        "scope": "validation-only calibration and error analysis",
        "random_forest_parameters": SELECTED_PARAMETERS,
        "calibration_selection": {
            "strategy": (
                "fit on earlier OOF seasons and evaluate on the next OOF season"
            ),
            "selection_metric": "mean_multiclass_log_loss",
            "tie_breaker": "mean_multiclass_brier_score",
            "oof_seasons": [
                fold.validation_season for fold in oof_folds
            ],
            "evaluation_seasons": [
                fold.validation_season for fold in oof_folds[1:]
            ],
            "oof_rows": int(len(calibration_target)),
            "candidates": candidates,
            "selected_method": selected["method"],
        },
        "train_seasons": sorted(training["season"].unique().tolist()),
        "validation_seasons": sorted(
            validation["season"].unique().tolist()
        ),
        "test_seasons_used": [],
        "holdout_seasons_used": [],
        "training_rows": int(len(training)),
        "validation_rows": int(len(validation)),
        "feature_count": len(columns),
        "validation_results": {
            "uncalibrated_random_forest": uncalibrated_metrics,
            "selected_calibration": calibrated_metrics,
        },
        "calibration_bins": calibration_bins,
        "error_analysis": analysis,
    }


def _metric_rows(report: dict[str, Any]) -> str:
    rows = []
    labels = {
        "uncalibrated_random_forest": "Uncalibrated Random Forest",
        "selected_calibration": (
            f"Selected calibration "
            f"({report['calibration_selection']['selected_method']})"
        ),
    }
    for key, label in labels.items():
        metrics = report["validation_results"][key]
        rows.append(
            f"| {label} | {metrics['accuracy']:.3f} | "
            f"{metrics['macro_f1']:.3f} | {metrics['log_loss']:.3f} | "
            f"{metrics['multiclass_brier_score']:.3f} | "
            f"{metrics['expected_calibration_error']:.3f} |"
        )
    return "\n".join(rows)


def _slice_rows(groups: dict[str, dict[str, Any]]) -> str:
    return "\n".join(
        f"| {name} | {metrics['rows']} | {metrics['accuracy']:.3f} | "
        f"{metrics['mean_confidence']:.3f} | {metrics['log_loss']:.3f} |"
        for name, metrics in groups.items()
    )


def render_calibration_report(report: dict[str, Any]) -> str:
    """Render calibration selection and validation diagnostics."""
    candidate_rows = []
    selected_method = report["calibration_selection"]["selected_method"]
    for candidate in report["calibration_selection"]["candidates"]:
        marker = " **selected**" if candidate["method"] == selected_method else ""
        candidate_rows.append(
            f"| {candidate['method']}{marker} | "
            f"{candidate['mean_log_loss']:.3f} | "
            f"{candidate['mean_brier_score']:.3f} | "
            f"{candidate['mean_expected_calibration_error']:.3f} | "
            f"{candidate['mean_macro_f1']:.3f} |"
        )

    errors = report["error_analysis"]
    mistake_rows = []
    for mistake in errors["highest_confidence_mistakes"]:
        mistake_rows.append(
            f"| {mistake['match_date']} | {mistake['home_team']} vs "
            f"{mistake['away_team']} | {mistake['actual']} | "
            f"{mistake['predicted']} | {mistake['confidence']:.3f} |"
        )
    if not mistake_rows:
        mistake_rows.append("| None | - | - | - | - |")

    return f"""# FootCast Phase 5 Calibration and Error-Analysis Report

**Status:** {str(report["status"]).upper()}

This checkpoint evaluates calibration for the selected Random Forest without
opening the 2024-25 test or 2025-26 holdout seasons. Base-model probabilities
are generated with expanding training-period folds. Each calibration method is
fitted on earlier out-of-fold seasons and evaluated on the next out-of-fold
season.

- Training rows: {report["training_rows"]}
- Validation rows: {report["validation_rows"]}
- Out-of-fold calibration rows: {report["calibration_selection"]["oof_rows"]}
- Calibration-selection evaluation seasons:
  {", ".join(report["calibration_selection"]["evaluation_seasons"])}
- Selected method: {selected_method}
- Test rows loaded: 0
- Holdout rows loaded: 0

## Training-period calibration selection

| Method | Mean log loss | Mean Brier | Mean ECE | Mean macro F1 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(candidate_rows)}

Log loss is the primary selection metric. Brier score breaks an exact tie.
Expected calibration error (ECE) summarizes the gap between confidence and
accuracy across confidence bins; lower values are better for all three
probability metrics.

## 2023-24 validation

| Model | Accuracy | Macro F1 | Log loss | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
{_metric_rows(report)}

Calibration adjusts probabilities rather than creating new match information.
It can improve reliability and log loss while leaving the most-likely class,
accuracy, or draw recall unchanged.

## Errors by actual outcome

| Slice | Rows | Accuracy | Mean confidence | Log loss |
| --- | ---: | ---: | ---: | ---: |
{_slice_rows(errors["by_actual_outcome"])}

## Errors by season timing

Early-season rows are matches where either team had played fewer than five
earlier league matches in that season.

| Slice | Rows | Accuracy | Mean confidence | Log loss |
| --- | ---: | ---: | ---: | ---: |
{_slice_rows(errors["by_season_timing"])}

## Errors by prior-history availability

Cold starts contain at least one team with no earlier observed Premier League
history in the dataset.

| Slice | Rows | Accuracy | Mean confidence | Log loss |
| --- | ---: | ---: | ---: | ---: |
{_slice_rows(errors["by_history"])}

## Errors by Elo gap

| Slice | Rows | Accuracy | Mean confidence | Log loss |
| --- | ---: | ---: | ---: | ---: |
{_slice_rows(errors["by_elo_gap"])}

## Highest-confidence mistakes

There were {errors["high_confidence_mistake_count"]} incorrect validation
predictions with at least 60% confidence. The table shows up to 15.

| Date | Match | Actual | Predicted | Confidence |
| --- | --- | --- | --- | ---: |
{chr(10).join(mistake_rows)}

Reliability plots show predicted probability against observed frequency for
each outcome; numbers next to points are bin sample sizes. Slice charts compare
accuracy with mean confidence. Small slices and sparse reliability bins should
not be over-interpreted.

This checkpoint does not freeze a production artifact or authorize test-season
evaluation. Those decisions follow review of these validation-only results.
"""


def run_calibration_pipeline(
    raw_dir: Path = DEFAULT_RAW_DIR,
    report_json: Path = DEFAULT_REPORT_JSON,
    report_markdown: Path = DEFAULT_REPORT_MARKDOWN,
    reliability_figure: Path = DEFAULT_RELIABILITY_FIGURE,
    error_figure: Path = DEFAULT_ERROR_FIGURE,
) -> dict[str, Any]:
    """Regenerate development data, calibration evidence, and diagnostics."""
    matches = load_match_statistics(raw_dir, splits=DEVELOPMENT_SPLITS)
    features = build_pre_match_features(matches)
    report = run_calibration_experiment(features)

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_markdown.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    report_markdown.write_text(
        render_calibration_report(report),
        encoding="utf-8",
    )
    plot_reliability_comparison(
        report["calibration_bins"]["uncalibrated"],
        report["calibration_bins"]["selected"],
        reliability_figure,
        selected_method=report["calibration_selection"]["selected_method"],
    )
    plot_error_slices(report["error_analysis"], error_figure)
    return report


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report = run_calibration_pipeline()
    metrics = report["validation_results"]["selected_calibration"]
    print(
        "Selected probability method="
        f"{report['calibration_selection']['selected_method']} "
        f"with validation log loss={metrics['log_loss']:.3f} "
        f"and ECE={metrics['expected_calibration_error']:.3f}."
    )


if __name__ == "__main__":
    main()
