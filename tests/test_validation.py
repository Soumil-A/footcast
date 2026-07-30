from datetime import date

import pandas as pd
import pytest

from footcast.data.manifest import DownloadSpec
from footcast.data.validate import (
    CANONICAL_COLUMNS,
    DataValidationError,
    validate_season,
)


@pytest.fixture
def spec() -> DownloadSpec:
    return DownloadSpec(
        season="2024-25",
        split="test",
        url="https://example.invalid/E0.csv",
        filename="premier_league_2024_25.csv",
        sha256="0" * 64,
        date_min=date(2024, 7, 1),
        date_max=date(2025, 7, 31),
        expected_rows=2,
        expected_teams=3,
    )


@pytest.fixture
def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Div": ["E0", "E0"],
            "Date": ["16/08/2024", "17/08/2024"],
            "HomeTeam": ["Arsenal", "Chelsea"],
            "AwayTeam": ["Chelsea", "Everton"],
            "FTHG": [2, 1],
            "FTAG": [0, 1],
            "FTR": ["H", "D"],
            "OptionalStatistic": [None, 4.0],
        }
    )


def test_valid_data_becomes_the_canonical_schema(
    valid_frame: pd.DataFrame, spec: DownloadSpec
) -> None:
    validated = validate_season(valid_frame, spec)

    assert tuple(validated.matches.columns) == CANONICAL_COLUMNS
    assert validated.matches["result"].tolist() == ["home_win", "draw"]
    assert validated.matches["match_date"].tolist() == [
        "2024-08-16",
        "2024-08-17",
    ]
    assert validated.missing_values == {"OptionalStatistic": 1}


def test_missing_required_columns_are_rejected(
    valid_frame: pd.DataFrame, spec: DownloadSpec
) -> None:
    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_season(valid_frame.drop(columns="FTR"), spec)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("Date", "not-a-date", "invalid dates"),
        ("HomeTeam", " ", "blank HomeTeam"),
        ("AwayTeam", "Chelsea ", "surrounding whitespace"),
        ("FTHG", -1, "nonnegative integer"),
        ("FTAG", 1.5, "nonnegative integer"),
        ("FTR", "X", "invalid FTR"),
    ],
)
def test_invalid_required_values_are_rejected(
    valid_frame: pd.DataFrame,
    spec: DownloadSpec,
    column: str,
    value: object,
    message: str,
) -> None:
    if column in {"FTHG", "FTAG"}:
        valid_frame[column] = valid_frame[column].astype("object")
    valid_frame.loc[0, column] = value

    with pytest.raises(DataValidationError, match=message):
        validate_season(valid_frame, spec)


def test_null_required_values_are_rejected(
    valid_frame: pd.DataFrame, spec: DownloadSpec
) -> None:
    valid_frame.loc[0, "HomeTeam"] = None

    with pytest.raises(DataValidationError, match="null required values"):
        validate_season(valid_frame, spec)


def test_a_team_cannot_play_itself(
    valid_frame: pd.DataFrame, spec: DownloadSpec
) -> None:
    valid_frame.loc[0, "AwayTeam"] = "Arsenal"

    with pytest.raises(DataValidationError, match="home and away teams are identical"):
        validate_season(valid_frame, spec)


def test_result_must_agree_with_goals(
    valid_frame: pd.DataFrame, spec: DownloadSpec
) -> None:
    valid_frame.loc[0, "FTR"] = "A"

    with pytest.raises(DataValidationError, match="disagrees with full-time goals"):
        validate_season(valid_frame, spec)


def test_duplicate_fixture_is_rejected(
    valid_frame: pd.DataFrame, spec: DownloadSpec
) -> None:
    duplicate = valid_frame.iloc[[0]].copy()
    frame = pd.concat([valid_frame, duplicate], ignore_index=True)
    larger_spec = DownloadSpec(
        **{
            **spec.__dict__,
            "expected_rows": 3,
        }
    )

    with pytest.raises(DataValidationError, match="duplicate fixtures"):
        validate_season(frame, larger_spec)


def test_date_outside_season_boundary_is_rejected(
    valid_frame: pd.DataFrame, spec: DownloadSpec
) -> None:
    valid_frame.loc[0, "Date"] = "01/08/2025"

    with pytest.raises(DataValidationError, match="dates outside"):
        validate_season(valid_frame, spec)


def test_row_and_team_counts_are_checked(
    valid_frame: pd.DataFrame, spec: DownloadSpec
) -> None:
    wrong_expectations = DownloadSpec(
        **{
            **spec.__dict__,
            "expected_rows": 380,
            "expected_teams": 20,
        }
    )

    with pytest.raises(DataValidationError) as error:
        validate_season(valid_frame, wrong_expectations)

    assert "expected 380 rows, found 2" in str(error.value)
    assert "expected 20 teams, found 3" in str(error.value)


def test_team_name_variants_are_rejected(
    valid_frame: pd.DataFrame, spec: DownloadSpec
) -> None:
    valid_frame.loc[1, "AwayTeam"] = "Ever ton"
    valid_frame.loc[0, "AwayTeam"] = "Everton"
    valid_frame.loc[0, "HomeTeam"] = "Ever ton"

    with pytest.raises(DataValidationError, match="inconsistent team-name variants"):
        validate_season(valid_frame, spec)
