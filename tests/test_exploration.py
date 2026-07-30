from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from footcast.data.manifest import DownloadSpec
from footcast.exploration import (
    FIGURE_FILENAMES,
    ExplorationDataset,
    generate_figures,
    load_exploration_data,
    outcome_distribution,
    promoted_performance,
    team_match_records,
    team_season_performance,
)


@pytest.fixture
def matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": ["2020-21", "2020-21", "2021-22", "2021-22"],
            "match_date": [
                "2020-08-01",
                "2020-08-08",
                "2021-08-01",
                "2021-08-08",
            ],
            "home_team": ["A", "B", "A", "C"],
            "away_team": ["B", "A", "C", "A"],
            "full_time_home_goals": [2, 1, 0, 1],
            "full_time_away_goals": [0, 1, 1, 1],
            "result": ["home_win", "draw", "away_win", "draw"],
            "home_shots": [10, 8, 7, 9],
            "away_shots": [5, 8, 11, 9],
            "home_shots_on_target": [5, 3, 2, 4],
            "away_shots_on_target": [2, 3, 5, 4],
            "home_fouls": [8, 9, 10, 11],
            "away_fouls": [10, 9, 8, 11],
            "home_corners": [6, 4, 3, 5],
            "away_corners": [2, 4, 7, 5],
            "home_yellow_cards": [1, 2, 3, 1],
            "away_yellow_cards": [2, 2, 1, 1],
            "home_red_cards": [0, 0, 1, 0],
            "away_red_cards": [0, 0, 0, 0],
        }
    )


@pytest.fixture
def promoted() -> dict[str, tuple[str, ...]]:
    return {"2020-21": (), "2021-22": ("C",)}


def test_outcome_distribution_is_percentage_by_season(
    matches: pd.DataFrame,
) -> None:
    distribution = outcome_distribution(matches)

    assert distribution.loc["2020-21", "home_win"] == 50
    assert distribution.loc["2020-21", "draw"] == 50
    assert distribution.loc["2021-22", "away_win"] == 50
    assert distribution.sum(axis=1).tolist() == [100, 100]


def test_team_records_assign_home_and_away_points(
    matches: pd.DataFrame,
) -> None:
    records = team_match_records(matches.iloc[[0]])

    assert records.loc[records["team"] == "A", "points"].item() == 3
    assert records.loc[records["team"] == "B", "points"].item() == 0
    assert records.loc[records["team"] == "A", "location"].item() == "home"
    assert records.loc[records["team"] == "B", "location"].item() == "away"


def test_team_season_summary_is_hand_calculated(
    matches: pd.DataFrame,
    promoted: dict[str, tuple[str, ...]],
) -> None:
    summary = team_season_performance(matches, promoted)
    team_a = summary[
        (summary["season"] == "2020-21") & (summary["team"] == "A")
    ].iloc[0]

    assert team_a["matches"] == 2
    assert team_a["points"] == 4
    assert team_a["points_per_match"] == 2
    assert team_a["goals_for"] == 3
    assert team_a["goals_against"] == 1


def test_new_team_comparison_uses_only_later_seasons(
    matches: pd.DataFrame,
    promoted: dict[str, tuple[str, ...]],
) -> None:
    comparison = promoted_performance(matches, promoted).set_index(
        "promoted_candidate"
    )

    assert comparison.loc[True, "matches"] == 2
    assert comparison.loc[False, "matches"] == 2
    assert comparison.loc[True, "points_per_match"] == 2
    assert comparison.loc[False, "points_per_match"] == pytest.approx(0.5)


def _raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["01/08/2020"],
            "HomeTeam": ["A"],
            "AwayTeam": ["B"],
            "FTHG": [1],
            "FTAG": [0],
            "FTR": ["H"],
            "HS": [7],
            "AS": [3],
            "HST": [3],
            "AST": [1],
            "HF": [8],
            "AF": [9],
            "HC": [4],
            "AC": [2],
            "HY": [1],
            "AY": [2],
            "HR": [0],
            "AR": [0],
        }
    )


def _spec(season: str, split: str, filename: str) -> DownloadSpec:
    start_year = int(season[:4])
    return DownloadSpec(
        season=season,
        split=split,
        url=f"https://example.invalid/{filename}",
        filename=filename,
        sha256="0" * 64,
        date_min=date(start_year, 7, 1),
        date_max=date(start_year + 1, 7, 31),
        expected_rows=1,
        expected_teams=2,
    )


def test_loader_excludes_test_and_holdout_without_reading_them(
    tmp_path: Path,
) -> None:
    train = _spec("2020-21", "train", "train.csv")
    holdout = _spec("2021-22", "holdout", "missing-holdout.csv")
    _raw_frame().to_csv(tmp_path / train.filename, index=False)

    dataset = load_exploration_data(tmp_path, specs=(train, holdout))

    assert dataset.matches["season"].tolist() == ["2020-21"]
    assert dataset.matches["split"].tolist() == ["train"]


def test_all_phase_two_figures_are_generated(
    tmp_path: Path,
    matches: pd.DataFrame,
    promoted: dict[str, tuple[str, ...]],
) -> None:
    missingness = pd.DataFrame(
        {
            "Date": [0.0, 0.0],
            "HomeTeam": [0.0, 0.0],
            "AwayTeam": [0.0, 0.0],
            "FTR": [0.0, 0.0],
            "HS": [0.0, 0.0],
            "Time": [100.0, 0.0],
        },
        index=["2020-21", "2021-22"],
    )
    dataset = ExplorationDataset(matches, missingness, promoted)

    paths = generate_figures(dataset, tmp_path)

    assert set(paths) == set(FIGURE_FILENAMES)
    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
