"""Run Phase 6 rolling backtests for Poisson and Dixon-Coles models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from footcast.config import DATA_SPLIT
from footcast.data.download import DEFAULT_RAW_DIR
from footcast.data.manifest import DEFAULT_MANIFEST
from footcast.data.matches import load_match_statistics
from footcast.evaluation.metrics import CLASS_LABELS, evaluate_predictions
from footcast.features.build_features import build_pre_match_features
from footcast.models.baselines import EloBaseline, LogisticRegressionBaseline
from footcast.models.goal_models import PoissonGoalModel
from footcast.models.random_forest import (
    SELECTED_PARAMETERS,
    aligned_probabilities,
    make_random_forest,
)
from footcast.models.run_baselines import model_feature_columns

if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"

PROJECT_ROOT = DEFAULT_MANIFEST.parents[1]
V2_DEVELOPMENT_SPLITS = frozenset({"train", "validation", "test"})
V2_SEASONS = DATA_SPLIT.train + DATA_SPLIT.validation + DATA_SPLIT.test
POISSON_ALPHAS = (0.1, 0.5, 1.0)
DIXON_COLES_RHOS = (-0.05, -0.10, -0.15)
DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports" / "goal_model_results.json"
DEFAULT_REPORT_MARKDOWN = PROJECT_ROOT / "reports" / "goal_model_results.md"
DEFAULT_COMPARISON_FIGURE = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "phase6"
    / "rolling_model_comparison.png"
)
DEFAULT_SEASON_FIGURE = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "phase6"
    / "log_loss_by_season.png"
)
DEFAULT_DRAW_FIGURE = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "phase6"
    / "draw_recall_by_model.png"
)
MODEL_ORDER = (
    "Elo",
    "Logistic regression",
    "Random Forest",
    "Poisson",
    "Dixon-Coles",
)


def rolling_season_folds(
    matches: pd.DataFrame,
    *,
    minimum_training_seasons: int = 3,
) -> list[dict[str, Any]]:
    """Create expanding v2 folds through the already-seen 2024-25 season."""
    if "split" not in matches or "season" not in matches:
        raise ValueError("Rolling backtests require split and season columns")
    if "holdout" in set(matches["split"]):
        raise ValueError("Holdout data entered v2 development")
    if set(matches["season"]) != set(V2_SEASONS):
        raise ValueError("V2 backtests require every season through 2024-25")
    if not 1 <= minimum_training_seasons < len(V2_SEASONS):
        raise ValueError("minimum_training_seasons must leave an evaluation fold")

    return [
        {
            "train_seasons": list(V2_SEASONS[:evaluation_index]),
            "evaluation_season": V2_SEASONS[evaluation_index],
        }
        for evaluation_index in range(
            minimum_training_seasons,
            len(V2_SEASONS),
        )
    ]


def _metrics(actual, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = np.asarray(CLASS_LABELS, dtype=object)[
        np.argmax(probabilities, axis=1)
    ]
    return evaluate_predictions(actual, predictions, probabilities)


def _candidate_summary(
    parameters: dict[str, Any],
    folds: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "parameters": parameters,
        "mean_log_loss": float(
            np.mean([fold["metrics"]["log_loss"] for fold in folds])
        ),
        "mean_macro_f1": float(
            np.mean([fold["metrics"]["macro_f1"] for fold in folds])
        ),
        "mean_draw_recall": float(
            np.mean(
                [
                    fold["metrics"]["per_class_recall"]["draw"]
                    for fold in folds
                ]
            )
        ),
        "folds": folds,
    }


def _select(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        candidates,
        key=lambda candidate: (
            candidate["mean_log_loss"],
            -candidate["mean_macro_f1"],
            str(candidate["parameters"]),
        ),
    )


def _aggregate_model(folds: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "accuracy",
        "macro_f1",
        "log_loss",
        "multiclass_brier_score",
        "expected_calibration_error",
    )
    aggregate = {
        f"mean_{metric}": float(
            np.mean([fold["metrics"][metric] for fold in folds])
        )
        for metric in metric_names
    }
    aggregate["mean_draw_recall"] = float(
        np.mean(
            [
                fold["metrics"]["per_class_recall"]["draw"]
                for fold in folds
            ]
        )
    )
    aggregate["folds"] = folds
    return aggregate


def run_goal_model_backtests(
    matches: pd.DataFrame,
    features: pd.DataFrame,
) -> dict[str, Any]:
    """Run fixed benchmarks and bounded goal-model research across seasons."""
    folds = rolling_season_folds(matches)
    feature_columns = model_feature_columns(features)
    poisson_records = {alpha: [] for alpha in POISSON_ALPHAS}
    dixon_records = {
        (alpha, rho): []
        for alpha in POISSON_ALPHAS
        for rho in DIXON_COLES_RHOS
    }
    benchmark_records = {
        "Elo": [],
        "Logistic regression": [],
        "Random Forest": [],
    }

    for fold in folds:
        train_seasons = fold["train_seasons"]
        evaluation_season = fold["evaluation_season"]
        train_matches = matches.loc[matches["season"].isin(train_seasons)]
        evaluation_matches = matches.loc[
            matches["season"] == evaluation_season
        ]
        train_features = features.loc[
            features["season"].isin(train_seasons)
        ]
        evaluation_features = features.loc[
            features["season"] == evaluation_season
        ]
        if len(evaluation_matches) != len(evaluation_features):
            raise ValueError("Match and feature rows differ in an evaluation fold")
        match_keys = evaluation_matches[
            ["season", "match_date", "home_team", "away_team"]
        ].reset_index(drop=True)
        feature_keys = evaluation_features[
            ["season", "match_date", "home_team", "away_team"]
        ].reset_index(drop=True)
        if not match_keys.equals(feature_keys):
            raise ValueError("Match and feature ordering differs")

        actual = evaluation_matches["result"].to_numpy(dtype=object)
        metadata = {
            "train_seasons": train_seasons,
            "evaluation_season": evaluation_season,
            "training_rows": int(len(train_matches)),
            "evaluation_rows": int(len(evaluation_matches)),
        }

        for alpha in POISSON_ALPHAS:
            goal_model = PoissonGoalModel(alpha=alpha).fit(train_matches)
            poisson_probabilities = goal_model.predict_proba(evaluation_matches)
            poisson_records[alpha].append(
                {**metadata, "metrics": _metrics(actual, poisson_probabilities)}
            )
            for rho in DIXON_COLES_RHOS:
                probabilities = goal_model.predict_proba(
                    evaluation_matches,
                    rho=rho,
                )
                dixon_records[(alpha, rho)].append(
                    {**metadata, "metrics": _metrics(actual, probabilities)}
                )

        x_train = train_features[feature_columns]
        y_train = train_features["result"]
        x_evaluation = evaluation_features[feature_columns]
        benchmark_models = {
            "Elo": EloBaseline(),
            "Logistic regression": LogisticRegressionBaseline(),
            "Random Forest": make_random_forest(SELECTED_PARAMETERS),
        }
        for name, model in benchmark_models.items():
            model.fit(x_train, y_train)
            probabilities = (
                aligned_probabilities(model, x_evaluation)
                if name == "Random Forest"
                else model.predict_proba(x_evaluation)
            )
            benchmark_records[name].append(
                {**metadata, "metrics": _metrics(actual, probabilities)}
            )

    poisson_candidates = [
        _candidate_summary({"alpha": alpha, "rho": 0.0}, records)
        for alpha, records in poisson_records.items()
    ]
    dixon_candidates = [
        _candidate_summary({"alpha": alpha, "rho": rho}, records)
        for (alpha, rho), records in dixon_records.items()
    ]
    selected_poisson = _select(poisson_candidates)
    selected_dixon = _select(dixon_candidates)
    models = {
        name: _aggregate_model(records)
        for name, records in benchmark_records.items()
    }
    models["Poisson"] = _aggregate_model(selected_poisson["folds"])
    models["Dixon-Coles"] = _aggregate_model(selected_dixon["folds"])

    return {
        "status": "passed",
        "phase": "Phase 6 v2 goal-model research",
        "evaluation_policy": (
            "expanding next-season development backtests; no final claim"
        ),
        "development_seasons": list(V2_SEASONS),
        "evaluation_seasons": [fold["evaluation_season"] for fold in folds],
        "holdout_seasons_used": [],
        "feature_count": len(feature_columns),
        "goal_model": {
            "formulation": (
                "regularized Poisson team attack and opponent defense with "
                "a home indicator"
            ),
            "maximum_score_grid": 10,
            "poisson_candidates": poisson_candidates,
            "dixon_coles_candidates": dixon_candidates,
            "selected_poisson": selected_poisson["parameters"],
            "selected_dixon_coles": selected_dixon["parameters"],
        },
        "models": models,
    }


def _comparison_rows(report: dict[str, Any]) -> str:
    rows = []
    for name in MODEL_ORDER:
        values = report["models"][name]
        rows.append(
            f"| {name} | {values['mean_accuracy']:.3f} | "
            f"{values['mean_macro_f1']:.3f} | "
            f"{values['mean_log_loss']:.3f} | "
            f"{values['mean_multiclass_brier_score']:.3f} | "
            f"{values['mean_draw_recall']:.3f} |"
        )
    return "\n".join(rows)


def _candidate_rows(candidates: list[dict[str, Any]], selected) -> str:
    rows = []
    for candidate in candidates:
        marker = " **selected**" if candidate["parameters"] == selected else ""
        rows.append(
            f"| {candidate['parameters']['alpha']:.2f} | "
            f"{candidate['parameters']['rho']:.2f}{marker} | "
            f"{candidate['mean_log_loss']:.3f} | "
            f"{candidate['mean_macro_f1']:.3f} | "
            f"{candidate['mean_draw_recall']:.3f} |"
        )
    return "\n".join(rows)


def render_goal_model_report(report: dict[str, Any]) -> str:
    """Render v2 backtest evidence without making a new final-test claim."""
    goal = report["goal_model"]
    return f"""# FootCast Phase 6 Goal-Model Backtest Report

**Status:** {str(report["status"]).upper()}

Phase 6 begins v2 research after the 2024-25 v1 test. Seasons through 2024-25
are now development data. Seven expanding folds train on earlier seasons and
evaluate the next season from 2018-19 through 2024-25. The 2025-26 holdout was
not loaded.

Poisson regression models each team's goals using a team attack effect, an
opponent defense effect, and a home indicator. Dixon-Coles adjusts the four
low-score cells (0-0, 0-1, 1-0, and 1-1), where football outcomes—especially
draws—often depart from independent Poisson assumptions.

## Goal-model settings

### Independent Poisson

| Alpha | Rho | Mean log loss | Mean macro F1 | Mean draw recall |
| ---: | ---: | ---: | ---: | ---: |
{_candidate_rows(goal["poisson_candidates"], goal["selected_poisson"])}

### Dixon-Coles

| Alpha | Rho | Mean log loss | Mean macro F1 | Mean draw recall |
| ---: | ---: | ---: | ---: | ---: |
{_candidate_rows(goal["dixon_coles_candidates"], goal["selected_dixon_coles"])}

Settings are compared as v2 development research on the same rolling folds.
These means are not a new final-test estimate.

## Rolling model comparison

| Model | Accuracy | Macro F1 | Log loss | Brier | Draw recall |
| --- | ---: | ---: | ---: | ---: | ---: |
{_comparison_rows(report)}

The charts show mean metrics, season-to-season log loss, and draw recall. A goal
model is useful only if its probability quality and draw behavior improve
without relying on the sealed holdout. Any v2 selection or feature expansion
must be decided before 2025-26 is opened.

## Conclusion

Neither goal model improved this checkpoint. Random Forest achieved the lowest
mean rolling log loss (`0.975`), with Elo effectively tied at `0.976`; Poisson
and Dixon-Coles reached `1.017` and `1.018`. The selected low-score correction
also did not make draws the most likely class for any evaluation match, so both
goal models had zero draw recall.

This is a negative but useful result. The current goal-rate formulation should
not replace the simpler benchmarks. The next experiment should be specified
before it runs and should target missing pre-match information or the decision
rule for draws—not repeatedly tune these results. The 2025-26 holdout remains
sealed.
"""


def run_goal_model_pipeline(
    raw_dir: Path = DEFAULT_RAW_DIR,
    report_json: Path = DEFAULT_REPORT_JSON,
    report_markdown: Path = DEFAULT_REPORT_MARKDOWN,
    comparison_figure: Path = DEFAULT_COMPARISON_FIGURE,
    season_figure: Path = DEFAULT_SEASON_FIGURE,
    draw_figure: Path = DEFAULT_DRAW_FIGURE,
) -> dict[str, Any]:
    """Run all v2 backtests and write reproducible evidence."""
    matches = load_match_statistics(
        raw_dir,
        splits=V2_DEVELOPMENT_SPLITS,
    )
    features = build_pre_match_features(matches)
    report = run_goal_model_backtests(matches, features)

    from footcast.evaluation.goal_model_plots import (
        plot_draw_recall,
        plot_goal_model_comparison,
        plot_log_loss_by_season,
    )

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_markdown.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report_markdown.write_text(render_goal_model_report(report), encoding="utf-8")
    plot_goal_model_comparison(report["models"], comparison_figure)
    plot_log_loss_by_season(report["models"], season_figure)
    plot_draw_recall(report["models"], draw_figure)
    return report


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report = run_goal_model_pipeline()
    best = min(
        report["models"],
        key=lambda name: report["models"][name]["mean_log_loss"],
    )
    print(
        f"Completed {len(report['evaluation_seasons'])} rolling backtests. "
        f"Best mean log loss: {best}."
    )


if __name__ == "__main__":
    main()
