from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from footcast.data.manifest import DownloadSpec
from footcast.data.matches import load_match_statistics
from footcast.features.build_features import (
    FeatureBuildError,
    build_pre_match_features,
)
from footcast.features.elo import EloConfig, expected_home_score


def _matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": ["2020-21", "2020-21", "2020-21"],
            "split": ["train", "train", "train"],
            "match_date": ["2020-08-01", "2020-08-08", "2020-08-15"],
            "home_team": ["A", "B", "A"],
            "away_team": ["B", "C", "C"],
            "full_time_home_goals": [2, 1, 0],
            "full_time_away_goals": [0, 1, 1],
            "result": ["home_win", "draw", "away_win"],
            "home_shots": [10, 7, 8],
            "away_shots": [4, 7, 9],
            "home_shots_on_target": [5, 3, 2],
            "away_shots_on_target": [1, 3, 4],
        }
    )


def test_first_match_cannot_see_current_or_future_information() -> None:
    features = build_pre_match_features(_matches())
    first = features.iloc[0]

    assert first["home_history_matches"] == 0
    assert first["away_history_matches"] == 0
    assert first["home_form_points_last_5"] == 0
    assert first["home_goals_for_last_5"] == 0
    assert first["home_shots_last_5"] == 0
    assert first["home_elo"] == 1500
    assert first["away_elo"] == 1500
    assert pd.isna(first["home_days_since_previous_match"])
    assert pd.isna(first["rest_days_difference"])
    assert "full_time_home_goals" not in features
    assert "home_shots" not in features


def test_second_match_uses_only_the_first_match() -> None:
    features = build_pre_match_features(_matches())
    second = features.iloc[1]

    assert second["home_team"] == "B"
    assert second["home_history_matches"] == 1
    assert second["home_form_points_last_5"] == 0
    assert second["home_goals_for_last_5"] == 0
    assert second["home_goals_against_last_5"] == 2
    assert second["home_shots_last_5"] == 4
    assert second["home_shots_on_target_last_5"] == 1
    assert second["home_days_since_previous_match"] == 7
    assert second["home_venue_matches_last_5"] == 0
    assert second["home_venue_points_last_5"] == 0
    assert second["away_history_matches"] == 0


def test_home_and_away_statistics_are_assigned_to_team_perspectives() -> None:
    features = build_pre_match_features(_matches())
    third = features.iloc[2]

    assert third["home_team"] == "A"
    assert third["home_goals_for_last_5"] == 2
    assert third["home_goals_against_last_5"] == 0
    assert third["home_shots_last_5"] == 10
    assert third["home_venue_matches_last_5"] == 1
    assert third["home_venue_points_last_5"] == 3
    assert third["home_expanding_goals_for_mean"] == 2
    assert third["home_expanding_goals_against_mean"] == 0
    assert third["away_team"] == "C"
    assert third["away_goals_for_last_5"] == 1
    assert third["away_goals_against_last_5"] == 1
    assert third["away_shots_last_5"] == 7
    assert third["away_expanding_goals_for_mean"] == 1
    assert third["away_expanding_goals_against_mean"] == 1


def test_elo_snapshot_precedes_the_current_result_update() -> None:
    config = EloConfig(home_advantage=0)
    features = build_pre_match_features(_matches(), elo_config=config)
    first = features.iloc[0]
    second = features.iloc[1]
    expected_change = config.k_factor * (
        1 - expected_home_score(1500, 1500, config)
    )

    assert first["home_elo"] == 1500
    assert first["away_elo"] == 1500
    assert second["home_team"] == "B"
    assert second["home_elo"] == pytest.approx(1500 - expected_change)
    assert second["away_elo"] == 1500


def test_rolling_window_excludes_current_match_and_keeps_only_five() -> None:
    rows = []
    for index in range(6):
        rows.append(
            {
                "season": "2020-21",
                "split": "train",
                "match_date": f"2020-08-{1 + index * 3:02d}",
                "home_team": "A",
                "away_team": f"Opponent {index}",
                "full_time_home_goals": index + 1,
                "full_time_away_goals": 0,
                "result": "home_win",
                "home_shots": 10 + index,
                "away_shots": 2,
                "home_shots_on_target": 3 + index,
                "away_shots_on_target": 1,
            }
        )
    features = build_pre_match_features(pd.DataFrame(rows))
    sixth = features.iloc[5]

    assert sixth["home_rolling_matches"] == 5
    assert sixth["home_form_points_last_5"] == 15
    assert sixth["home_form_wins_last_5"] == 5
    assert sixth["home_goals_for_last_5"] == 1 + 2 + 3 + 4 + 5
    assert sixth["home_shots_last_5"] == 10 + 11 + 12 + 13 + 14
    assert sixth["home_goals_for_last_5"] != sum(range(1, 7))


def test_season_state_resets_while_general_history_carries_forward() -> None:
    matches = _matches().iloc[[0, 2]].copy()
    matches.loc[matches.index[1], "season"] = "2021-22"
    matches.loc[matches.index[1], "match_date"] = "2021-08-01"
    features = build_pre_match_features(matches)
    second_season = features.iloc[1]

    assert second_season["home_history_matches"] == 1
    assert second_season["home_form_points_last_5"] == 3
    assert second_season["home_season_matches_played"] == 0
    assert second_season["home_season_points"] == 0


def test_matchup_differences_are_home_minus_away() -> None:
    features = build_pre_match_features(_matches())
    third = features.iloc[2]

    assert third["form_points_difference"] == 2
    assert third["goals_scored_difference"] == 1
    assert third["goals_conceded_difference"] == -1
    assert third["rest_days_difference"] == 7
    assert third["shots_on_target_difference"] == 2


def test_missing_required_statistics_stop_feature_building() -> None:
    matches = _matches().drop(columns="home_shots")

    with pytest.raises(FeatureBuildError, match="Missing required match columns"):
        build_pre_match_features(matches)


def test_duplicate_fixture_stops_feature_building() -> None:
    matches = pd.concat([_matches(), _matches().iloc[[0]]], ignore_index=True)

    with pytest.raises(FeatureBuildError, match="Duplicate fixtures"):
        build_pre_match_features(matches)


def _raw_match() -> pd.DataFrame:
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


def test_match_loader_does_not_read_unrequested_holdout(
    tmp_path: Path,
) -> None:
    train = _spec("2020-21", "train", "train.csv")
    holdout = _spec("2021-22", "holdout", "missing.csv")
    _raw_match().to_csv(tmp_path / train.filename, index=False)

    loaded = load_match_statistics(
        tmp_path, specs=(train, holdout), splits=frozenset({"train"})
    )

    assert loaded["season"].tolist() == ["2020-21"]
    assert loaded["split"].tolist() == ["train"]
