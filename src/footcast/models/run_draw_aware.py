"""Run Phase 6 draw-aware backtests and the checkpoint 3 decision gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from footcast.data.download import DEFAULT_RAW_DIR
from footcast.data.manifest import DEFAULT_MANIFEST
from footcast.data.matches import load_match_statistics
from footcast.evaluation.metrics import CLASS_LABELS, evaluate_predictions
from footcast.features.build_features import build_pre_match_features
from footcast.models.baselines import EloBaseline
from footcast.models.draw_aware import TwoStageDrawClassifier
from footcast.models.random_forest import (
    SELECTED_PARAMETERS,
    aligned_probabilities,
    make_random_forest,
)
from footcast.models.run_baselines import model_feature_columns
from footcast.models.run_goal_models import (
    V2_DEVELOPMENT_SPLITS,
    V2_SEASONS,
    rolling_season_folds,
)

if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"

PROJECT_ROOT = DEFAULT_MANIFEST.parents[1]
DRAW_WEIGHTS = (1.0, 1.25, 1.5, 2.0)
ACCEPTANCE_THRESHOLDS = {
    "maximum_mean_log_loss": 0.970,
    "maximum_mean_brier_score": 0.578,
    "minimum_mean_macro_f1": 0.400,
    "minimum_mean_draw_recall": 0.100,
}
DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports" / "draw_aware_results.json"
DEFAULT_REPORT_MARKDOWN = PROJECT_ROOT / "reports" / "draw_aware_results.md"
DEFAULT_COMPARISON_FIGURE = (
    PROJECT_ROOT / "reports" / "figures" / "phase6" / "draw_aware_comparison.png"
)
DEFAULT_SEASON_FIGURE = (
    PROJECT_ROOT / "reports" / "figures" / "phase6" / "draw_aware_by_season.png"
)
DEFAULT_TRADEOFF_FIGURE = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "phase6"
    / "draw_weight_tradeoff.png"
)


def _metrics(actual: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = np.asarray(CLASS_LABELS, dtype=object)[
        np.argmax(probabilities, axis=1)
    ]
    return evaluate_predictions(actual, predictions, probabilities)


def _aggregate(folds: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "accuracy",
        "macro_f1",
        "log_loss",
        "multiclass_brier_score",
        "expected_calibration_error",
    )
    result = {
        f"mean_{metric}": float(
            np.mean([fold["metrics"][metric] for fold in folds])
        )
        for metric in metrics
    }
    result["mean_draw_recall"] = float(
        np.mean(
            [fold["metrics"]["per_class_recall"]["draw"] for fold in folds]
        )
    )
    result["folds"] = folds
    return result


def select_draw_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Select log loss first, with macro F1 and lower weight as tie-breakers."""
    if not candidates:
        raise ValueError("At least one draw-aware candidate is required")
    return min(
        candidates,
        key=lambda candidate: (
            candidate["mean_log_loss"],
            -candidate["mean_macro_f1"],
            candidate["parameters"]["draw_weight"],
        ),
    )


def checkpoint_decision(metrics: dict[str, float]) -> dict[str, Any]:
    """Apply the fixed checkpoint 3 gate without adapting thresholds."""
    required = {
        "mean_log_loss",
        "mean_multiclass_brier_score",
        "mean_macro_f1",
        "mean_draw_recall",
    }
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError(f"Decision metrics are missing: {missing}")
    checks = {
        "mean_log_loss_at_most_0.970": (
            metrics["mean_log_loss"]
            <= ACCEPTANCE_THRESHOLDS["maximum_mean_log_loss"]
        ),
        "mean_brier_at_most_0.578": (
            metrics["mean_multiclass_brier_score"]
            <= ACCEPTANCE_THRESHOLDS["maximum_mean_brier_score"]
        ),
        "mean_macro_f1_at_least_0.400": (
            metrics["mean_macro_f1"]
            >= ACCEPTANCE_THRESHOLDS["minimum_mean_macro_f1"]
        ),
        "mean_draw_recall_at_least_0.100": (
            metrics["mean_draw_recall"]
            >= ACCEPTANCE_THRESHOLDS["minimum_mean_draw_recall"]
        ),
    }
    promoted = all(checks.values())
    return {
        "promoted": promoted,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "phase7_reference_model": (
            "Two-stage draw-aware logistic regression" if promoted else "Elo"
        ),
        "phase7_use": (
            "educational probability demonstration with documented limitations"
        ),
    }


def run_draw_aware_backtests(
    matches: pd.DataFrame,
    features: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate fixed draw weights and references on expanding season folds."""
    folds = rolling_season_folds(matches)
    feature_columns = model_feature_columns(features)
    candidate_records = {weight: [] for weight in DRAW_WEIGHTS}
    reference_records: dict[str, list[dict[str, Any]]] = {
        "Elo": [],
        "Random Forest": [],
    }

    for fold in folds:
        train_seasons = fold["train_seasons"]
        evaluation_season = fold["evaluation_season"]
        train = features.loc[features["season"].isin(train_seasons)]
        evaluation = features.loc[features["season"] == evaluation_season]
        if train.empty or evaluation.empty:
            raise ValueError("Every rolling fold requires train and evaluation rows")
        x_train = train[feature_columns]
        y_train = train["result"]
        x_evaluation = evaluation[feature_columns]
        actual = evaluation["result"].to_numpy(dtype=object)
        metadata = {
            "train_seasons": train_seasons,
            "evaluation_season": evaluation_season,
            "training_rows": int(len(train)),
            "evaluation_rows": int(len(evaluation)),
        }

        for weight in DRAW_WEIGHTS:
            model = TwoStageDrawClassifier(draw_weight=weight).fit(
                x_train,
                y_train,
            )
            candidate_records[weight].append(
                {
                    **metadata,
                    "metrics": _metrics(actual, model.predict_proba(x_evaluation)),
                }
            )

        elo = EloBaseline().fit(x_train, y_train)
        forest = make_random_forest(SELECTED_PARAMETERS).fit(x_train, y_train)
        reference_records["Elo"].append(
            {
                **metadata,
                "metrics": _metrics(actual, elo.predict_proba(x_evaluation)),
            }
        )
        reference_records["Random Forest"].append(
            {
                **metadata,
                "metrics": _metrics(
                    actual,
                    aligned_probabilities(forest, x_evaluation),
                ),
            }
        )

    candidates = []
    for weight, records in candidate_records.items():
        candidate = _aggregate(records)
        candidate["parameters"] = {"draw_weight": weight}
        candidates.append(candidate)
    selected = select_draw_candidate(candidates)
    references = {
        name: _aggregate(records) for name, records in reference_records.items()
    }
    decision = checkpoint_decision(selected)
    models = {
        **references,
        "Two-stage draw-aware": selected,
    }
    return {
        "status": "passed",
        "phase": "Phase 6 checkpoints 2 and 3",
        "development_seasons": list(V2_SEASONS),
        "evaluation_seasons": [fold["evaluation_season"] for fold in folds],
        "holdout_seasons_used": [],
        "feature_count": len(feature_columns),
        "candidate_selection_rule": (
            "lowest mean rolling log loss; macro F1 and lower draw weight "
            "break exact ties"
        ),
        "acceptance_thresholds": ACCEPTANCE_THRESHOLDS,
        "candidates": candidates,
        "selected_parameters": selected["parameters"],
        "models": models,
        "decision": decision,
    }


def _candidate_rows(report: dict[str, Any]) -> str:
    rows = []
    selected = report["selected_parameters"]
    for candidate in report["candidates"]:
        marker = " **selected**" if candidate["parameters"] == selected else ""
        rows.append(
            f"| {candidate['parameters']['draw_weight']:.2f}{marker} | "
            f"{candidate['mean_log_loss']:.3f} | "
            f"{candidate['mean_multiclass_brier_score']:.3f} | "
            f"{candidate['mean_macro_f1']:.3f} | "
            f"{candidate['mean_draw_recall']:.3f} |"
        )
    return "\n".join(rows)


def _model_rows(report: dict[str, Any]) -> str:
    rows = []
    for name, model in report["models"].items():
        rows.append(
            f"| {name} | {model['mean_log_loss']:.3f} | "
            f"{model['mean_multiclass_brier_score']:.3f} | "
            f"{model['mean_macro_f1']:.3f} | "
            f"{model['mean_draw_recall']:.3f} |"
        )
    return "\n".join(rows)


def _gate_rows(report: dict[str, Any]) -> str:
    return "\n".join(
        f"| `{name}` | {'PASS' if passed else 'FAIL'} |"
        for name, passed in report["decision"]["checks"].items()
    )


def render_draw_aware_report(report: dict[str, Any]) -> str:
    """Render checkpoint evidence and the automated promote/reject decision."""
    decision = report["decision"]
    outcome = "PROMOTE" if decision["promoted"] else "REJECT"
    return f"""# FootCast Phase 6 Draw-Aware Decision Report

**Pipeline status:** {report['status'].upper()}
**Checkpoint 3 decision:** {outcome}

Checkpoint 2 tests a two-stage logistic model. The first classifier estimates
draw versus non-draw; the second estimates home versus away conditional on a
decisive match. Their probabilities are combined into the fixed order home win,
draw, and away win. Four draw weights were fixed before the results were run.

All seven backtests train on earlier seasons and evaluate the next season from
2018-19 through 2024-25. The 2025-26 holdout was not loaded.

## Checkpoint 2 candidate search

| Draw weight | Log loss | Brier | Macro F1 | Draw recall |
| ---: | ---: | ---: | ---: | ---: |
{_candidate_rows(report)}

## Reference comparison

| Model | Log loss | Brier | Macro F1 | Draw recall |
| --- | ---: | ---: | ---: | ---: |
{_model_rows(report)}

## Checkpoint 3 fixed acceptance gate

| Criterion | Result |
| --- | --- |
{_gate_rows(report)}

The gate requires every criterion to pass. The decision is **{outcome}**.
The Phase 7 reference model is **{decision['phase7_reference_model']}**, used
only as an {decision['phase7_use']}.

Increasing draw weight may improve draw recall while worsening probability
quality. The candidate table and tradeoff chart preserve that evidence rather
than treating one label metric as sufficient. This decision does not open or
make any claim about the sealed 2025-26 holdout.
"""


def run_draw_aware_pipeline(
    raw_dir: Path = DEFAULT_RAW_DIR,
    report_json: Path = DEFAULT_REPORT_JSON,
    report_markdown: Path = DEFAULT_REPORT_MARKDOWN,
    comparison_figure: Path = DEFAULT_COMPARISON_FIGURE,
    season_figure: Path = DEFAULT_SEASON_FIGURE,
    tradeoff_figure: Path = DEFAULT_TRADEOFF_FIGURE,
) -> dict[str, Any]:
    """Run checkpoint 2 and write the checkpoint 3 decision evidence."""
    matches = load_match_statistics(raw_dir, splits=V2_DEVELOPMENT_SPLITS)
    features = build_pre_match_features(matches)
    report = run_draw_aware_backtests(matches, features)

    from footcast.evaluation.draw_aware_plots import (
        plot_draw_aware_by_season,
        plot_draw_aware_comparison,
        plot_draw_weight_tradeoff,
    )

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_markdown.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report_markdown.write_text(render_draw_aware_report(report), encoding="utf-8")
    plot_draw_aware_comparison(report["models"], comparison_figure)
    plot_draw_aware_by_season(report["models"], season_figure)
    plot_draw_weight_tradeoff(report["candidates"], tradeoff_figure)
    return report


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report = run_draw_aware_pipeline()
    decision = "promoted" if report["decision"]["promoted"] else "rejected"
    print(
        "Completed Phase 6 draw-aware backtests. "
        f"Candidate {decision}; Phase 7 reference: "
        f"{report['decision']['phase7_reference_model']}."
    )


if __name__ == "__main__":
    main()
