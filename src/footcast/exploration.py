"""Reproducible Phase 2 exploratory analysis with explicit leakage labels."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "footcast-matplotlib")
)

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from footcast.data.download import DEFAULT_RAW_DIR
from footcast.data.manifest import DEFAULT_MANIFEST, DownloadSpec, load_manifest
from footcast.data.matches import prepare_match_statistics

PROJECT_ROOT = DEFAULT_MANIFEST.parents[1]
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "phase2"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "exploration.md"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "reports" / "exploration_summary.json"
DEVELOPMENT_SPLITS = frozenset({"train", "validation"})

OUTCOME_ORDER = ("home_win", "draw", "away_win")
OUTCOME_LABELS = {
    "home_win": "Home win",
    "draw": "Draw",
    "away_win": "Away win",
}
FIGURE_FILENAMES = {
    "outcomes": "outcome_distribution_by_season.png",
    "home_advantage": "home_win_percentage_by_season.png",
    "team_performance": "team_season_ppg_heatmap.png",
    "head_to_head": "home_vs_away_points_heatmap.png",
    "correlations": "current_match_correlation_heatmap.png",
    "missingness": "source_missingness_heatmap.png",
    "promoted": "promoted_team_performance.png",
}


@dataclass(frozen=True)
class ExplorationDataset:
    """Validated development matches and source-level missingness metadata."""

    matches: pd.DataFrame
    source_missingness: pd.DataFrame
    promoted_candidates: dict[str, tuple[str, ...]]


def load_exploration_data(
    raw_dir: Path = DEFAULT_RAW_DIR,
    *,
    specs: tuple[DownloadSpec, ...] | None = None,
    splits: frozenset[str] = DEVELOPMENT_SPLITS,
) -> ExplorationDataset:
    """Load only approved splits and validate every source file before analysis."""
    selected = tuple(
        spec
        for spec in (specs or load_manifest())
        if spec.split in splits
    )
    if not selected:
        raise ValueError("No manifest seasons match the requested exploration splits")

    raw_frames: list[pd.DataFrame] = []
    matches: list[pd.DataFrame] = []
    teams_by_season: dict[str, set[str]] = {}
    for spec in selected:
        path = raw_dir / spec.filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `python -m footcast.data.pipeline` first."
            )
        frame = pd.read_csv(path)
        raw_frames.append(frame)
        prepared = prepare_match_statistics(frame, spec)
        matches.append(prepared)
        teams_by_season[spec.season] = set(prepared["home_team"]) | set(
            prepared["away_team"]
        )

    all_source_columns = sorted(
        set().union(*(set(frame.columns) for frame in raw_frames))
    )
    missing_rows: list[dict[str, Any]] = []
    for spec, frame in zip(selected, raw_frames, strict=True):
        row: dict[str, Any] = {"season": spec.season}
        for column in all_source_columns:
            row[column] = (
                float(frame[column].isna().mean() * 100)
                if column in frame
                else 100.0
            )
        missing_rows.append(row)

    promoted: dict[str, tuple[str, ...]] = {}
    previous_teams: set[str] | None = None
    for spec in selected:
        current_teams = teams_by_season[spec.season]
        promoted[spec.season] = (
            tuple(sorted(current_teams - previous_teams))
            if previous_teams is not None
            else ()
        )
        previous_teams = current_teams

    return ExplorationDataset(
        matches=pd.concat(matches, ignore_index=True).sort_values(
            ["match_date", "home_team", "away_team"], ignore_index=True
        ),
        source_missingness=pd.DataFrame(missing_rows).set_index("season"),
        promoted_candidates=promoted,
    )


def outcome_distribution(matches: pd.DataFrame) -> pd.DataFrame:
    """Return outcome percentages by season in chronological column order."""
    counts = pd.crosstab(matches["season"], matches["result"])
    counts = counts.reindex(columns=OUTCOME_ORDER, fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0).mul(100)


def team_match_records(
    matches: pd.DataFrame,
    promoted_candidates: dict[str, tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    """Convert each fixture into home- and away-team descriptive records."""
    home_points = matches["result"].map(
        {"home_win": 3, "draw": 1, "away_win": 0}
    )
    away_points = matches["result"].map(
        {"home_win": 0, "draw": 1, "away_win": 3}
    )
    home = pd.DataFrame(
        {
            "season": matches["season"],
            "match_date": matches["match_date"],
            "team": matches["home_team"],
            "location": "home",
            "points": home_points,
            "goals_for": matches["full_time_home_goals"],
            "goals_against": matches["full_time_away_goals"],
        }
    )
    away = pd.DataFrame(
        {
            "season": matches["season"],
            "match_date": matches["match_date"],
            "team": matches["away_team"],
            "location": "away",
            "points": away_points,
            "goals_for": matches["full_time_away_goals"],
            "goals_against": matches["full_time_home_goals"],
        }
    )
    records = pd.concat([home, away], ignore_index=True)
    promoted_candidates = promoted_candidates or {}
    records["promoted_candidate"] = [
        team in promoted_candidates.get(season, ())
        for season, team in zip(
            records["season"], records["team"], strict=True
        )
    ]
    return records


def team_season_performance(
    matches: pd.DataFrame,
    promoted_candidates: dict[str, tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    """Summarize points and goal difference per match for each team-season."""
    records = team_match_records(matches, promoted_candidates)
    summary = (
        records.groupby(["season", "team"], as_index=False)
        .agg(
            matches=("points", "size"),
            points=("points", "sum"),
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
            promoted_candidate=("promoted_candidate", "max"),
        )
        .sort_values(["season", "team"], ignore_index=True)
    )
    summary["points_per_match"] = summary["points"] / summary["matches"]
    summary["goal_difference_per_match"] = (
        summary["goals_for"] - summary["goals_against"]
    ) / summary["matches"]
    return summary


def head_to_head_home_points(
    matches: pd.DataFrame, *, team_limit: int = 12
) -> pd.DataFrame:
    """Return average home points for the most frequently observed teams."""
    appearances = pd.concat(
        [matches["home_team"], matches["away_team"]]
    ).value_counts()
    teams = list(appearances.head(team_limit).index)
    selected = matches[
        matches["home_team"].isin(teams) & matches["away_team"].isin(teams)
    ].copy()
    selected["home_points"] = selected["result"].map(
        {"home_win": 3, "draw": 1, "away_win": 0}
    )
    matrix = selected.pivot_table(
        index="home_team",
        columns="away_team",
        values="home_points",
        aggfunc="mean",
    )
    return matrix.reindex(index=teams, columns=teams)


def correlation_frame(matches: pd.DataFrame) -> pd.DataFrame:
    """Build compact home-oriented current-match quantities for correlation."""
    home_points = matches["result"].map(
        {"home_win": 3, "draw": 1, "away_win": 0}
    )
    return pd.DataFrame(
        {
            "home_result_points": home_points,
            "goal_difference": (
                matches["full_time_home_goals"]
                - matches["full_time_away_goals"]
            ),
            "shot_difference": matches["home_shots"] - matches["away_shots"],
            "shots_on_target_difference": (
                matches["home_shots_on_target"]
                - matches["away_shots_on_target"]
            ),
            "corner_difference": (
                matches["home_corners"] - matches["away_corners"]
            ),
            "foul_difference": matches["home_fouls"] - matches["away_fouls"],
            "yellow_card_difference": (
                matches["home_yellow_cards"] - matches["away_yellow_cards"]
            ),
            "red_card_difference": (
                matches["home_red_cards"] - matches["away_red_cards"]
            ),
        }
    )


def promoted_performance(
    matches: pd.DataFrame,
    promoted_candidates: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    """Compare descriptive match averages for new and returning team names."""
    records = team_match_records(matches, promoted_candidates)
    records = records[records["season"] != matches["season"].min()]
    summary = (
        records.groupby("promoted_candidate", as_index=False)
        .agg(
            matches=("points", "size"),
            points_per_match=("points", "mean"),
            goals_for_per_match=("goals_for", "mean"),
            goals_against_per_match=("goals_against", "mean"),
        )
        .sort_values("promoted_candidate", ascending=False, ignore_index=True)
    )
    summary["group"] = summary["promoted_candidate"].map(
        {True: "New to league dataset", False: "Continuing team"}
    )
    return summary


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F8FAFC",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
        }
    )


def select_missingness_columns(missingness: pd.DataFrame) -> list[str]:
    """Select stable core fields plus informative optional drift columns."""
    core = [
        column
        for column in (
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTR",
            "HS",
            "HST",
            "HC",
            "HY",
        )
        if column in missingness
    ]
    variability = missingness.nunique().sort_values(ascending=False)
    optional = [
        column
        for column in variability.index
        if column not in core and variability[column] > 1
    ][:16]
    return core + optional


def generate_figures(
    dataset: ExplorationDataset, output_dir: Path = DEFAULT_FIGURE_DIR
) -> dict[str, Path]:
    """Generate all Phase 2 figures and return their paths."""
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        key: output_dir / filename for key, filename in FIGURE_FILENAMES.items()
    }
    matches = dataset.matches

    outcomes = outcome_distribution(matches)
    outcomes.rename(columns=OUTCOME_LABELS).plot(
        kind="bar",
        stacked=True,
        color=["#2563EB", "#94A3B8", "#F97316"],
        figsize=(10, 5.5),
    )
    plt.title("Premier League outcome distribution by season")
    plt.xlabel("Season")
    plt.ylabel("Matches (%)")
    plt.legend(
        title="Result",
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=3,
    )
    plt.xticks(rotation=35, ha="right")
    _save_figure(paths["outcomes"])

    home_wins = outcomes["home_win"]
    plt.figure(figsize=(9.5, 4.8))
    sns.lineplot(
        x=home_wins.index,
        y=home_wins.values,
        marker="o",
        color="#2563EB",
        linewidth=2.5,
    )
    plt.axhline(
        home_wins.mean(),
        color="#64748B",
        linestyle="--",
        label=f"Development average ({home_wins.mean():.1f}%)",
    )
    plt.title("Home-win percentage over development seasons")
    plt.xlabel("Season")
    plt.ylabel("Home wins (%)")
    plt.xticks(rotation=35, ha="right")
    plt.legend()
    _save_figure(paths["home_advantage"])

    team_summary = team_season_performance(
        matches, dataset.promoted_candidates
    )
    frequent_teams = (
        team_summary.groupby("team")["season"]
        .nunique()
        .sort_values(ascending=False)
        .head(16)
        .index
    )
    team_matrix = team_summary[
        team_summary["team"].isin(frequent_teams)
    ].pivot(index="team", columns="season", values="points_per_match")
    plt.figure(figsize=(11, 7))
    sns.heatmap(
        team_matrix,
        cmap="RdYlBu",
        center=1.4,
        vmin=0.5,
        vmax=2.5,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        cbar_kws={"label": "Points per match"},
    )
    plt.title("Team performance by season (16 most observed teams)")
    plt.xlabel("Season")
    plt.ylabel("Team")
    _save_figure(paths["team_performance"])

    head_to_head = head_to_head_home_points(matches)
    plt.figure(figsize=(10, 8.5))
    sns.heatmap(
        head_to_head,
        cmap="RdYlBu",
        center=1.5,
        vmin=0,
        vmax=3,
        annot=True,
        fmt=".1f",
        linewidths=0.3,
        cbar_kws={"label": "Average home points"},
    )
    plt.title("Historical home points by matchup (12 most observed teams)")
    plt.xlabel("Away team")
    plt.ylabel("Home team")
    _save_figure(paths["head_to_head"])

    correlations = correlation_frame(matches).corr()
    plt.figure(figsize=(9, 7.5))
    sns.heatmap(
        correlations,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={"label": "Pearson correlation"},
    )
    plt.title("Current-match statistics correlation")
    _save_figure(paths["correlations"])

    missing_columns = select_missingness_columns(dataset.source_missingness)
    plt.figure(figsize=(13, 5.5))
    sns.heatmap(
        dataset.source_missingness[missing_columns],
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        linewidths=0.3,
        cbar_kws={"label": "Missing or absent (%)"},
    )
    plt.title("Source missingness and schema drift")
    plt.xlabel("Football-Data source column")
    plt.ylabel("Season")
    plt.xticks(rotation=55, ha="right")
    _save_figure(paths["missingness"])

    promoted = promoted_performance(matches, dataset.promoted_candidates)
    plt.figure(figsize=(7.5, 4.8))
    sns.barplot(
        data=promoted,
        x="group",
        y="points_per_match",
        hue="group",
        palette=["#F97316", "#2563EB"],
        legend=False,
    )
    plt.title("Performance of new vs continuing team names")
    plt.xlabel("")
    plt.ylabel("Points per match")
    plt.ylim(0, max(1.8, promoted["points_per_match"].max() + 0.15))
    _save_figure(paths["promoted"])
    return paths


def build_summary(dataset: ExplorationDataset) -> dict[str, Any]:
    """Calculate concise, JSON-safe descriptive findings."""
    matches = dataset.matches
    outcomes = outcome_distribution(matches)
    overall = (
        matches["result"]
        .value_counts(normalize=True)
        .reindex(OUTCOME_ORDER)
        .mul(100)
    )
    home_wins = outcomes["home_win"]
    team_summary = team_season_performance(
        matches, dataset.promoted_candidates
    )
    promoted = promoted_performance(matches, dataset.promoted_candidates)
    correlations = correlation_frame(matches).corr()["home_result_points"]
    correlation_candidates = correlations.drop(
        ["home_result_points", "goal_difference"]
    )
    strongest_name = correlation_candidates.abs().idxmax()
    strongest_value = float(correlations[strongest_name])
    top_team = (
        team_match_records(matches)
        .groupby("team")["points"]
        .agg(["sum", "count"])
        .assign(points_per_match=lambda frame: frame["sum"] / frame["count"])
        .sort_values("points_per_match", ascending=False)
        .iloc[0]
    )
    top_team_name = (
        team_match_records(matches)
        .groupby("team")["points"]
        .agg(["sum", "count"])
        .assign(points_per_match=lambda frame: frame["sum"] / frame["count"])
        .sort_values("points_per_match", ascending=False)
        .index[0]
    )
    promoted_rows = promoted.set_index("promoted_candidate")
    return {
        "scope": {
            "splits": sorted(DEVELOPMENT_SPLITS),
            "first_season": matches["season"].min(),
            "last_season": matches["season"].max(),
            "matches": int(len(matches)),
            "excluded_splits": ["test", "holdout"],
        },
        "outcomes_percent": {
            outcome: round(float(overall[outcome]), 2) for outcome in OUTCOME_ORDER
        },
        "home_win_range": {
            "lowest_season": str(home_wins.idxmin()),
            "lowest_percent": round(float(home_wins.min()), 2),
            "highest_season": str(home_wins.idxmax()),
            "highest_percent": round(float(home_wins.max()), 2),
            "average_percent": round(float(home_wins.mean()), 2),
        },
        "top_team_overall": {
            "team": str(top_team_name),
            "points_per_match": round(float(top_team["points_per_match"]), 3),
        },
        "promoted_comparison": {
            "new_team_points_per_match": round(
                float(promoted_rows.loc[True, "points_per_match"]), 3
            ),
            "continuing_team_points_per_match": round(
                float(promoted_rows.loc[False, "points_per_match"]), 3
            ),
        },
        "strongest_non_goal_current_match_correlation": {
            "variable": str(strongest_name),
            "correlation_with_home_result_points": round(strongest_value, 3),
        },
        "team_seasons": int(len(team_summary)),
    }


def render_report(summary: dict[str, Any]) -> str:
    """Render findings and the required interpretation contract."""
    outcomes = summary["outcomes_percent"]
    home = summary["home_win_range"]
    promoted = summary["promoted_comparison"]
    correlation = summary["strongest_non_goal_current_match_correlation"]
    top_team = summary["top_team_overall"]
    return f"""# FootCast Phase 2: Exploratory Analysis

## Scope and evaluation guard

This report uses {summary["scope"]["matches"]} validated matches from
{summary["scope"]["first_season"]} through {summary["scope"]["last_season"]}.
Only training and validation seasons are explored. The 2024-25 test season and
2025-26 holdout remain excluded from all Phase 2 calculations and figures.

No chart is a model evaluation, and no model or predictive feature is created.

## Main descriptive findings

- Home wins account for {outcomes["home_win"]:.2f}% of development matches,
  draws {outcomes["draw"]:.2f}%, and away wins
  {outcomes["away_win"]:.2f}%.
- Home-win frequency ranges from {home["lowest_percent"]:.2f}% in
  {home["lowest_season"]} to {home["highest_percent"]:.2f}% in
  {home["highest_season"]}. The development-season average is
  {home["average_percent"]:.2f}%.
- {top_team["team"]} has the highest overall points-per-match value in this
  development window ({top_team["points_per_match"]:.3f}). This is descriptive
  historical performance, not proof of future strength.
- Teams newly appearing relative to the prior season average
  {promoted["new_team_points_per_match"]:.3f} points per match versus
  {promoted["continuing_team_points_per_match"]:.3f} for continuing teams.
  "New" is a dataset-derived promotion/relegation candidate label, not an
  independently verified league-status field.
- Among non-goal current-match quantities, `{correlation["variable"]}` has the
  largest absolute Pearson correlation with home-result points
  ({correlation["correlation_with_home_result_points"]:.3f}). Correlation does
  not establish causation or make the variable safe before kickoff.

## Figure interpretation guide

### Outcome distribution by season

![Outcome distribution](figures/phase2/{FIGURE_FILENAMES["outcomes"]})

- **Question:** How does the home/draw/away balance vary by season?
- **Available before kickoff?** No. Each outcome is known only after the match.
- **Use:** Exploratory and useful for later class-balance decisions.
- **Do not conclude:** A past seasonal rate is a probability for a future
  individual fixture.

### Home-win percentage over time

![Home-win percentage](figures/phase2/{FIGURE_FILENAMES["home_advantage"]})

- **Question:** Has aggregate home advantage been stable?
- **Available before kickoff?** Historical rates are available; the current
  match result is not.
- **Use:** Exploratory. A lagged historical rate could be proposed and tested in
  Phase 3, but this same-season full rate cannot be used directly.
- **Do not conclude:** Changes are caused by one factor or will continue.

### Team-by-season performance heatmap

![Team-season performance](figures/phase2/{FIGURE_FILENAMES["team_performance"]})

- **Question:** Which frequently observed teams accumulated the most points per
  match in each season?
- **Available before kickoff?** Completed-match history is available; final
  season aggregates are not available during that season.
- **Use:** Exploratory. Phase 3 may create shifted, date-specific equivalents.
- **Do not conclude:** Retrospective season strength is a leakage-safe feature.

### Home-team versus away-team history

![Head-to-head home points](figures/phase2/{FIGURE_FILENAMES["head_to_head"]})

- **Question:** How have the most frequently observed team pairings differed
  when one side played at home?
- **Available before kickoff?** Only results completed before a future kickoff.
- **Use:** Exploratory; the matrix shown uses the complete development window.
- **Do not conclude:** Sparse head-to-head averages are stable or causal.

### Numerical correlation heatmap

![Current-match correlations](figures/phase2/{FIGURE_FILENAMES["correlations"]})

- **Question:** Which current-match numerical differences move together?
- **Available before kickoff?** No. Goals, shots, corners, fouls, and cards in
  the current match are post-kickoff information.
- **Use:** Descriptive only. Phase 3 may use shifted histories of these fields.
- **Do not conclude:** Correlation implies causation or predictive availability.

### Missing-value and schema-drift heatmap

![Source missingness](figures/phase2/{FIGURE_FILENAMES["missingness"]})

- **Question:** Which stable and representative optional source columns are
  absent or incomplete across seasons?
- **Available before kickoff?** This is dataset metadata, not a match feature.
- **Use:** Guides later column selection and missing-data handling.
- **Do not conclude:** A missing optional bookmaker field invalidates the
  canonical match row.

### New-team comparison

![Promoted-team comparison](figures/phase2/{FIGURE_FILENAMES["promoted"]})

- **Question:** Do teams newly appearing in the league dataset perform
  differently from continuing teams?
- **Available before kickoff?** Promotion status is generally known pre-season,
  but this label is inferred only from adjacent dataset team lists.
- **Use:** Exploratory until an explicit, verified promotion-status source or
  rule is approved.
- **Do not conclude:** The observed gap is caused by promotion alone.

## Reproduction

```bash
python -m footcast.exploration
```

The command revalidates the selected raw seasons, regenerates all figures, and
writes this report plus `reports/exploration_summary.json`.
"""


def run_exploration(
    raw_dir: Path = DEFAULT_RAW_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
) -> dict[str, Any]:
    """Run the complete Phase 2 analysis and write reproducible artifacts."""
    dataset = load_exploration_data(raw_dir)
    generate_figures(dataset, figure_dir)
    summary = build_summary(dataset)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary), encoding="utf-8")
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    summary = run_exploration()
    print(
        f"Explored {summary['scope']['matches']} development matches from "
        f"{summary['scope']['first_season']} through "
        f"{summary['scope']['last_season']}."
    )


if __name__ == "__main__":
    main()
