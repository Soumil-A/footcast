"""Build a development-only table of leakage-safe pre-match features."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from footcast.data.download import DEFAULT_RAW_DIR
from footcast.data.manifest import DEFAULT_MANIFEST
from footcast.data.matches import load_match_statistics
from footcast.features.elo import EloConfig, EloRatings
from footcast.features.form import CompletedTeamMatch, TeamHistory

PROJECT_ROOT = DEFAULT_MANIFEST.parents[1]
DEFAULT_FEATURE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "pre_match_features.csv"
)
DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports" / "feature_quality.json"
DEFAULT_REPORT_MARKDOWN = PROJECT_ROOT / "reports" / "feature_quality.md"
DEVELOPMENT_SPLITS = frozenset({"train", "validation"})

REQUIRED_MATCH_COLUMNS = frozenset(
    {
        "season",
        "split",
        "match_date",
        "home_team",
        "away_team",
        "full_time_home_goals",
        "full_time_away_goals",
        "result",
        "home_shots",
        "away_shots",
        "home_shots_on_target",
        "away_shots_on_target",
    }
)
TEAM_FEATURES = (
    "history_matches",
    "rolling_matches",
    "form_points_last_5",
    "form_wins_last_5",
    "goals_for_last_5",
    "goals_against_last_5",
    "shots_last_5",
    "shots_on_target_last_5",
    "venue_matches_last_5",
    "venue_points_last_5",
    "days_since_previous_match",
    "season_matches_played",
    "season_points",
    "expanding_goals_for_mean",
    "expanding_goals_against_mean",
    "elo",
)
DIFFERENCE_FEATURES = (
    "elo_difference",
    "form_points_difference",
    "goals_scored_difference",
    "goals_conceded_difference",
    "rest_days_difference",
    "shots_on_target_difference",
)
IDENTIFIER_COLUMNS = (
    "season",
    "split",
    "match_date",
    "home_team",
    "away_team",
)


@dataclass(frozen=True)
class FeatureDefinition:
    """One documented feature family and its leakage contract."""

    name: str
    definition: str
    source_columns: str
    available: str
    leakage_risk: str
    missing_history: str


FEATURE_DEFINITIONS = (
    FeatureDefinition(
        "history_matches / rolling_matches",
        "All prior observed matches / count used in the five-match window",
        "Date, HomeTeam, AwayTeam",
        "Immediately before kickoff",
        "Counting the current match",
        "Zero; the match row is retained",
    ),
    FeatureDefinition(
        "form_points_last_5 / form_wins_last_5",
        "Points and wins summed over up to five completed prior matches",
        "FTR",
        "After each prior match reaches full time",
        "Current result included before snapshot",
        "Zero with rolling_matches indicating sample size",
    ),
    FeatureDefinition(
        "goals_for_last_5 / goals_against_last_5",
        "Goals summed over up to five completed prior matches",
        "FTHG, FTAG",
        "After each prior match reaches full time",
        "Current score included",
        "Zero with rolling_matches indicating sample size",
    ),
    FeatureDefinition(
        "shots_last_5 / shots_on_target_last_5",
        "Team shots summed over up to five completed prior matches",
        "HS, AS, HST, AST",
        "After each prior match reaches full time",
        "Current-match statistics included",
        "Zero with rolling_matches indicating sample size",
    ),
    FeatureDefinition(
        "venue_points_last_5",
        "Points in up to five prior matches at the same home/away role",
        "HomeTeam, AwayTeam, FTR",
        "After each same-role prior match",
        "Mixing current result or wrong team perspective",
        "Zero with venue_matches_last_5 indicating sample size",
    ),
    FeatureDefinition(
        "days_since_previous_match",
        "Calendar days since the team's last observed completed match",
        "Date",
        "At the scheduled match date",
        "Incorrect ordering or future date",
        "Missing for the first observed match; row retained",
    ),
    FeatureDefinition(
        "season_matches_played / season_points",
        "Completed matches and points earlier in the same season",
        "Season, FTR",
        "After each prior same-season match",
        "Using final table totals",
        "Zero at the start of every season",
    ),
    FeatureDefinition(
        "expanding_goals_for_mean / expanding_goals_against_mean",
        "Mean goals across all completed prior observed matches",
        "FTHG, FTAG",
        "After each prior match reaches full time",
        "Current/future goals included",
        "Zero with history_matches indicating sample size",
    ),
    FeatureDefinition(
        "elo",
        "Rating snapshot before kickoff; result update happens afterward",
        "HomeTeam, AwayTeam, FTR",
        "Immediately before kickoff",
        "Updating from the current result too early",
        "1500 for an unseen team",
    ),
    FeatureDefinition(
        "*_difference",
        "Home pre-match value minus away pre-match value",
        "Corresponding home and away pre-match features",
        "Immediately before kickoff",
        "Either side contains current-match data",
        "Rest difference is missing if either rest value is missing",
    ),
)


class FeatureBuildError(ValueError):
    """Raised when source matches cannot safely produce features."""


def _team_result_values(result: str) -> tuple[int, int, int, int]:
    """Return home points/win and away points/win."""
    mapping = {
        "home_win": (3, 1, 0, 0),
        "draw": (1, 0, 1, 0),
        "away_win": (0, 0, 3, 1),
    }
    try:
        return mapping[result]
    except KeyError as error:
        raise FeatureBuildError(f"Unsupported result: {result}") from error


def _prefix(values: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{name}": value for name, value in values.items()}


def _difference(home: Any, away: Any) -> float | None:
    if pd.isna(home) or pd.isna(away):
        return None
    return float(home) - float(away)


def build_pre_match_features(
    matches: pd.DataFrame,
    *,
    elo_config: EloConfig | None = None,
) -> pd.DataFrame:
    """Create feature snapshots, then update histories after each result."""
    missing = sorted(REQUIRED_MATCH_COLUMNS - set(matches.columns))
    if missing:
        raise FeatureBuildError(f"Missing required match columns: {missing}")
    if matches.empty:
        raise FeatureBuildError("Cannot build features from an empty match table")

    ordered = matches.copy()
    ordered["_parsed_date"] = pd.to_datetime(
        ordered["match_date"], errors="coerce"
    )
    if ordered["_parsed_date"].isna().any():
        raise FeatureBuildError("Every match_date must be parseable")
    stat_columns = [
        "home_shots",
        "away_shots",
        "home_shots_on_target",
        "away_shots_on_target",
    ]
    if ordered[stat_columns].isna().any().any():
        raise FeatureBuildError("Required historical match statistics cannot be null")
    if ordered.duplicated(
        ["season", "match_date", "home_team", "away_team"]
    ).any():
        raise FeatureBuildError("Duplicate fixtures cannot produce unique features")
    ordered = ordered.sort_values(
        ["_parsed_date", "home_team", "away_team"], ignore_index=True
    )

    histories: dict[str, TeamHistory] = {}
    elo = EloRatings(elo_config or EloConfig())
    rows: list[dict[str, Any]] = []

    for match in ordered.to_dict(orient="records"):
        match_date: date = match["_parsed_date"].date()
        season = str(match["season"])
        home_team = str(match["home_team"])
        away_team = str(match["away_team"])
        result = str(match["result"])
        home_history = histories.setdefault(home_team, TeamHistory())
        away_history = histories.setdefault(away_team, TeamHistory())

        home_snapshot = home_history.snapshot(
            match_date=match_date, season=season, location="home"
        )
        away_snapshot = away_history.snapshot(
            match_date=match_date, season=season, location="away"
        )
        home_snapshot["elo"] = elo.get(home_team)
        away_snapshot["elo"] = elo.get(away_team)

        row: dict[str, Any] = {
            column: match[column] for column in IDENTIFIER_COLUMNS
        }
        row.update(_prefix(home_snapshot, "home"))
        row.update(_prefix(away_snapshot, "away"))
        row.update(
            {
                "elo_difference": _difference(
                    home_snapshot["elo"], away_snapshot["elo"]
                ),
                "form_points_difference": _difference(
                    home_snapshot["form_points_last_5"],
                    away_snapshot["form_points_last_5"],
                ),
                "goals_scored_difference": _difference(
                    home_snapshot["goals_for_last_5"],
                    away_snapshot["goals_for_last_5"],
                ),
                "goals_conceded_difference": _difference(
                    home_snapshot["goals_against_last_5"],
                    away_snapshot["goals_against_last_5"],
                ),
                "rest_days_difference": _difference(
                    home_snapshot["days_since_previous_match"],
                    away_snapshot["days_since_previous_match"],
                ),
                "shots_on_target_difference": _difference(
                    home_snapshot["shots_on_target_last_5"],
                    away_snapshot["shots_on_target_last_5"],
                ),
                "result": result,
            }
        )
        rows.append(row)

        home_points, home_win, away_points, away_win = _team_result_values(
            result
        )
        home_history.record(
            CompletedTeamMatch(
                match_date=match_date,
                season=season,
                location="home",
                points=home_points,
                win=home_win,
                goals_for=int(match["full_time_home_goals"]),
                goals_against=int(match["full_time_away_goals"]),
                shots=float(match["home_shots"]),
                shots_on_target=float(match["home_shots_on_target"]),
            )
        )
        away_history.record(
            CompletedTeamMatch(
                match_date=match_date,
                season=season,
                location="away",
                points=away_points,
                win=away_win,
                goals_for=int(match["full_time_away_goals"]),
                goals_against=int(match["full_time_home_goals"]),
                shots=float(match["away_shots"]),
                shots_on_target=float(match["away_shots_on_target"]),
            )
        )
        elo.update(home_team, away_team, result)

    return pd.DataFrame(rows)


def build_feature_report(features: pd.DataFrame) -> dict[str, Any]:
    """Summarize cold starts, missing rest, and output coverage."""
    feature_columns = [
        column
        for column in features.columns
        if column not in {*IDENTIFIER_COLUMNS, "result"}
    ]
    allowed_missing = {
        "home_days_since_previous_match",
        "away_days_since_previous_match",
        "rest_days_difference",
    }
    unexpected_missing = {
        column: int(count)
        for column, count in features.isna().sum().items()
        if count > 0 and column not in allowed_missing
    }
    return {
        "status": "passed",
        "rows": int(len(features)),
        "seasons": sorted(features["season"].unique().tolist()),
        "splits": sorted(features["split"].unique().tolist()),
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "rows_by_season": {
            str(season): int(count)
            for season, count in features.groupby("season").size().items()
        },
        "max_rolling_matches": int(
            features[
                ["home_rolling_matches", "away_rolling_matches"]
            ].max().max()
        ),
        "cold_start_rows": int(
            (
                (features["home_history_matches"] == 0)
                | (features["away_history_matches"] == 0)
            ).sum()
        ),
        "missing_rest_rows": int(features["rest_days_difference"].isna().sum()),
        "test_or_holdout_rows": int(
            features["split"].isin({"test", "holdout"}).sum()
        ),
        "unexpected_missing_values": unexpected_missing,
        "elo": {
            "initial_rating": 1500.0,
            "k_factor": 20.0,
            "home_advantage": 65.0,
        },
    }


def render_feature_report(report: dict[str, Any]) -> str:
    """Render a concise, reviewable feature-quality report."""
    return f"""# FootCast Phase 3 Feature-Quality Report

**Status:** {str(report["status"]).upper()}

- Rows: {report["rows"]}
- Seasons: {", ".join(report["seasons"])}
- Splits: {", ".join(report["splits"])}
- Pre-match feature columns: {report["feature_count"]}
- Rows where at least one team has no observed history: {report["cold_start_rows"]}
- Rows with an unavailable rest-days difference: {report["missing_rest_rows"]}
- Test or holdout rows: {report["test_or_holdout_rows"]}
- Maximum rolling-window matches: {report["max_rolling_matches"]}
- Unexpected missing values: {report["unexpected_missing_values"]}

All rows are retained. Rolling windows contain at most five completed prior
matches. Season counters reset at each new season, while general form and Elo
carry forward from earlier completed matches.

Elo uses an initial rating of {report["elo"]["initial_rating"]:.0f},
`K={report["elo"]["k_factor"]:.0f}`, and a
{report["elo"]["home_advantage"]:.0f}-point home adjustment when calculating
the expected result. Each row records the rating before that match; updates
happen only after the result is recorded.

The generated table contains development splits only. The 2024-25 test season
and 2025-26 holdout remain excluded.
"""


def run_feature_pipeline(
    raw_dir: Path = DEFAULT_RAW_DIR,
    feature_path: Path = DEFAULT_FEATURE_PATH,
    report_json: Path = DEFAULT_REPORT_JSON,
    report_markdown: Path = DEFAULT_REPORT_MARKDOWN,
) -> dict[str, Any]:
    """Build development features and write local and reviewable artifacts."""
    matches = load_match_statistics(
        raw_dir, splits=DEVELOPMENT_SPLITS
    )
    features = build_pre_match_features(matches)
    report = build_feature_report(features)
    if report["test_or_holdout_rows"] != 0:
        raise FeatureBuildError("Test and holdout rows entered development features")
    if report["max_rolling_matches"] > 5:
        raise FeatureBuildError("A rolling window contains more than five matches")
    if report["unexpected_missing_values"]:
        raise FeatureBuildError(
            "Unexpected missing values entered the feature table"
        )

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_markdown.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(feature_path, index=False)
    report_json.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    report_markdown.write_text(
        render_feature_report(report), encoding="utf-8"
    )
    return report


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report = run_feature_pipeline()
    print(
        f"Built {report['feature_count']} pre-match features for "
        f"{report['rows']} development matches."
    )


if __name__ == "__main__":
    main()
