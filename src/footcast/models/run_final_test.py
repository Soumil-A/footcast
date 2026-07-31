"""Run the frozen FootCast v1 model's one-time 2024-25 test evaluation."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from footcast.config import DATA_SPLIT
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
from footcast.evaluation.plots import (
    plot_model_comparison,
    plot_single_confusion_matrix,
)
from footcast.features.build_features import (
    IDENTIFIER_COLUMNS,
    build_pre_match_features,
)
from footcast.models.baselines import (
    AlwaysHomeBaseline,
    EloBaseline,
    LogisticRegressionBaseline,
    MajorityClassBaseline,
)
from footcast.models.frozen import (
    FROZEN_FEATURE_COLUMNS,
    MODEL_VERSION,
    fit_frozen_model,
    frozen_specification,
    save_frozen_artifact,
    specification_sha256,
)
from footcast.models.random_forest import aligned_probabilities

PROJECT_ROOT = DEFAULT_MANIFEST.parents[1]
FINAL_EVALUATION_SPLITS = frozenset({"train", "validation", "test"})
DEFAULT_SPEC_PATH = PROJECT_ROOT / "models" / "frozen_model_spec.json"
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "models" / f"{MODEL_VERSION}.joblib"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports" / "final_test_results.json"
DEFAULT_REPORT_MARKDOWN = PROJECT_ROOT / "reports" / "final_test_results.md"
DEFAULT_CONFUSION_FIGURE = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "phase5"
    / "final_test_confusion_matrix.png"
)
DEFAULT_COMPARISON_FIGURE = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "phase5"
    / "final_test_model_comparison.png"
)
DEFAULT_RELIABILITY_FIGURE = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "phase5"
    / "final_test_reliability.png"
)
DEFAULT_ERROR_FIGURE = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "phase5"
    / "final_test_error_slices.png"
)
MODEL_NAMES = (
    "Majority class",
    "Always home",
    "Elo",
    "Logistic regression",
    "Frozen Random Forest",
)


def verify_frozen_specification(path: Path = DEFAULT_SPEC_PATH) -> None:
    """Stop if the reviewed JSON contract differs from executable constants."""
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen model specification: {path}")
    reviewed = json.loads(path.read_text(encoding="utf-8"))
    if reviewed != frozen_specification():
        raise ValueError("Frozen model JSON and executable contract differ")


def split_final_evaluation_features(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enforce development fitting, one test season, and zero holdout rows."""
    required = {*IDENTIFIER_COLUMNS, "result", *FROZEN_FEATURE_COLUMNS}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Final evaluation is missing columns: {missing}")
    observed_splits = set(features["split"])
    if "holdout" in observed_splits:
        raise ValueError("Holdout data entered final test evaluation")
    if observed_splits != FINAL_EVALUATION_SPLITS:
        raise ValueError(
            "Final evaluation requires exactly train, validation, and test"
        )

    development = features.loc[
        features["split"].isin({"train", "validation"})
    ].copy()
    test = features.loc[features["split"] == "test"].copy()
    expected_fit_seasons = set(DATA_SPLIT.train + DATA_SPLIT.validation)
    if set(development["season"]) != expected_fit_seasons:
        raise ValueError("Frozen fit seasons do not match the contract")
    if set(test["season"]) != set(DATA_SPLIT.test):
        raise ValueError("Frozen test season does not match the contract")
    if development["match_date"].max() >= test["match_date"].min():
        raise ValueError("Every fit match must precede the test season")

    feature_columns = [
        column
        for column in features.columns
        if column not in {*IDENTIFIER_COLUMNS, "result"}
    ]
    if tuple(feature_columns) != FROZEN_FEATURE_COLUMNS:
        raise ValueError("Generated features differ from the frozen feature order")
    return development, test


def _predictions(probabilities: np.ndarray) -> np.ndarray:
    return np.asarray(CLASS_LABELS, dtype=object)[
        np.argmax(probabilities, axis=1)
    ]


def evaluate_frozen_models(features: pd.DataFrame) -> tuple[dict[str, Any], Any]:
    """Fit fixed models on development and score the test exactly once."""
    development, test = split_final_evaluation_features(features)
    columns = list(FROZEN_FEATURE_COLUMNS)
    x_fit = development[columns]
    y_fit = development["result"]
    x_test = test[columns]
    y_test = test["result"]

    models = {
        "Majority class": MajorityClassBaseline(),
        "Always home": AlwaysHomeBaseline(),
        "Elo": EloBaseline(),
        "Logistic regression": LogisticRegressionBaseline(),
    }
    results = {}
    for name, model in models.items():
        model.fit(x_fit, y_fit)
        probabilities = model.predict_proba(x_test)
        results[name] = evaluate_predictions(
            y_test,
            model.predict(x_test),
            probabilities,
        )

    forest = fit_frozen_model(x_fit, y_fit)
    forest_probabilities = aligned_probabilities(forest, x_test)
    results["Frozen Random Forest"] = evaluate_predictions(
        y_test,
        _predictions(forest_probabilities),
        forest_probabilities,
    )
    analysis = build_error_analysis(test, forest_probabilities)
    calibration_bins = classwise_calibration_bins(
        y_test,
        forest_probabilities,
    )
    report = {
        "status": "passed",
        "checkpoint": "Phase 5 checkpoint 2",
        "test_evaluation_status": "opened_after_model_contract_freeze",
        "model_version": MODEL_VERSION,
        "specification_sha256": specification_sha256(),
        "frozen_specification": frozen_specification(),
        "fit_seasons": list(DATA_SPLIT.train + DATA_SPLIT.validation),
        "test_seasons": list(DATA_SPLIT.test),
        "holdout_seasons_used": [],
        "fit_rows": int(len(development)),
        "test_rows": int(len(test)),
        "feature_count": len(columns),
        "results": results,
        "model_decision": {
            "status": "not_recommended_for_deployment",
            "reasons": [
                "Test accuracy and macro F1 fell below validation results.",
                "The frozen forest did not outperform both simpler baselines.",
                "Draw precision and recall are both zero.",
            ],
            "test_result_used_for_retuning": False,
        },
        "calibration_bins": calibration_bins,
        "error_analysis": analysis,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    return report, forest


def _result_rows(report: dict[str, Any]) -> str:
    rows = []
    for name in MODEL_NAMES:
        metrics = report["results"][name]
        rows.append(
            f"| {name} | {metrics['accuracy']:.3f} | "
            f"{metrics['macro_f1']:.3f} | {metrics['log_loss']:.3f} | "
            f"{metrics['multiclass_brier_score']:.3f} | "
            f"{metrics['expected_calibration_error']:.3f} |"
        )
    return "\n".join(rows)


def _class_rows(metrics: dict[str, Any]) -> str:
    return "\n".join(
        f"| {label} | {metrics['per_class_precision'][label]:.3f} | "
        f"{metrics['per_class_recall'][label]:.3f} |"
        for label in CLASS_LABELS
    )


def _validation_comparison_rows(forest: dict[str, Any]) -> str:
    reference = {
        "Accuracy": (0.5789473684210527, forest["accuracy"]),
        "Macro F1": (0.425, forest["macro_f1"]),
        "Log loss": (0.931, forest["log_loss"]),
        "Brier score": (0.547, forest["multiclass_brier_score"]),
        "ECE": (0.049, forest["expected_calibration_error"]),
    }
    return "\n".join(
        f"| {name} | {validation_value:.3f} | {test_value:.3f} |"
        for name, (validation_value, test_value) in reference.items()
    )


def _error_highlights(report: dict[str, Any]) -> str:
    forest = report["results"]["Frozen Random Forest"]
    analysis = report["error_analysis"]
    values = (
        ("Home-win recall", forest["per_class_recall"]["home_win"]),
        ("Draw recall", forest["per_class_recall"]["draw"]),
        ("Away-win recall", forest["per_class_recall"]["away_win"]),
        (
            "High-confidence mistakes at or above 60%",
            analysis["high_confidence_mistake_count"],
        ),
        (
            "Close Elo-gap accuracy",
            analysis["by_elo_gap"]["close_0_50"]["accuracy"],
        ),
        (
            "Large Elo-gap accuracy",
            analysis["by_elo_gap"]["large_over_100"]["accuracy"],
        ),
    )
    return "\n".join(
        f"- {name}: {value if isinstance(value, int) else f'{value:.3f}'}"
        for name, value in values
    )


def render_final_test_report(report: dict[str, Any]) -> str:
    """Render the frozen model's unbiased test result."""
    forest = report["results"]["Frozen Random Forest"]
    return f"""# FootCast Phase 5 Final Test Report

**Status:** {str(report["status"]).upper()}

The model contract was frozen before the 2024-25 test season was loaded. The
frozen Random Forest uses 38 pre-match features, 300 trees, depth 6, minimum
leaf size 20, no class weighting, and no probability calibration. It was
refitted on 2015-16 through 2023-24, then evaluated on 2024-25 without changing
features, preprocessing, parameters, thresholds, or calibration.

- Model version: `{report["model_version"]}`
- Specification SHA-256: `{report["specification_sha256"]}`
- Fit rows: {report["fit_rows"]}
- Test rows: {report["test_rows"]}
- Test season: {", ".join(report["test_seasons"])}
- Holdout rows loaded: 0

Test-season features remain pre-match. A match may use results from earlier
completed matches in 2024-25, but never its own result or a later result.

## Final 2024-25 model comparison

| Model | Accuracy | Macro F1 | Log loss | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
{_result_rows(report)}

## Frozen Random Forest class metrics

| Outcome | Precision | Recall |
| --- | ---: | ---: |
{_class_rows(forest)}

## Model decision

**Not recommended for deployment.** The frozen forest's test accuracy and
macro F1 fell below its validation results. Logistic regression achieved higher
test accuracy and macro F1, while Elo achieved lower log loss, lower Brier
score, and lower ECE. The frozen forest also selected no draws correctly.

This decision does not replace or retune v1 after seeing the test. It records
that the frozen candidate did not meet the standard for deployment.

## Validation-to-test context

| Metric | 2023-24 validation | 2024-25 test |
| --- | ---: | ---: |
{_validation_comparison_rows(forest)}

## Final error-analysis highlights

{_error_highlights(report)}

The test result is the unbiased estimate for the frozen v1 development process.
It must not be used to revise v1 and then reported again as if still unseen.
Any future model changes create a new candidate and require a new evaluation
policy. The 2025-26 holdout remains sealed for a later demonstration.
"""


def run_final_test_pipeline(
    raw_dir: Path = DEFAULT_RAW_DIR,
    report_json: Path = DEFAULT_REPORT_JSON,
    report_markdown: Path = DEFAULT_REPORT_MARKDOWN,
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    confusion_figure: Path = DEFAULT_CONFUSION_FIGURE,
    comparison_figure: Path = DEFAULT_COMPARISON_FIGURE,
    reliability_figure: Path = DEFAULT_RELIABILITY_FIGURE,
    error_figure: Path = DEFAULT_ERROR_FIGURE,
) -> dict[str, Any]:
    """Run the frozen one-time test and write immutable evidence artifacts."""
    verify_frozen_specification()
    matches = load_match_statistics(
        raw_dir,
        splits=FINAL_EVALUATION_SPLITS,
    )
    features = build_pre_match_features(matches)
    report, forest = evaluate_frozen_models(features)
    artifact = save_frozen_artifact(forest, artifact_path)
    artifact["path"] = f"models/{artifact_path.name}"
    report["local_model_artifact"] = artifact

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_markdown.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    report_markdown.write_text(
        render_final_test_report(report),
        encoding="utf-8",
    )
    forest_metrics = report["results"]["Frozen Random Forest"]
    plot_single_confusion_matrix(
        forest_metrics,
        confusion_figure,
        title="Frozen Random Forest — 2024-25 final test",
    )
    plot_model_comparison(
        report["results"],
        comparison_figure,
        title="FootCast frozen model comparison — 2024-25 final test",
    )
    plot_reliability_comparison(
        report["calibration_bins"],
        report["calibration_bins"],
        reliability_figure,
        selected_method="uncalibrated frozen v1",
        evaluation_label="2024-25 final test",
    )
    plot_error_slices(
        report["error_analysis"],
        error_figure,
        title="Frozen v1 error slices — 2024-25 final test",
    )
    return report


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report = run_final_test_pipeline()
    metrics = report["results"]["Frozen Random Forest"]
    print(
        f"Frozen {report['model_version']} test: "
        f"accuracy={metrics['accuracy']:.3f}, "
        f"macro F1={metrics['macro_f1']:.3f}, "
        f"log loss={metrics['log_loss']:.3f}."
    )


if __name__ == "__main__":
    main()
