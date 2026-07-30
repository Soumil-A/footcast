"""Strict validation and canonicalization of one Premier League season."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from footcast.data.manifest import DownloadSpec

REQUIRED_SOURCE_COLUMNS = frozenset(
    {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
)
CANONICAL_COLUMNS = (
    "season",
    "match_date",
    "home_team",
    "away_team",
    "full_time_home_goals",
    "full_time_away_goals",
    "result",
)
RESULT_LABELS = {"H": "home_win", "D": "draw", "A": "away_win"}


class DataValidationError(ValueError):
    """Raised with all detected blocking data problems."""

    def __init__(self, season: str, problems: Iterable[str]) -> None:
        self.season = season
        self.problems = tuple(problems)
        details = "\n".join(f"- {problem}" for problem in self.problems)
        super().__init__(f"Validation failed for {season}:\n{details}")


@dataclass(frozen=True)
class ValidatedSeason:
    """Canonical matches plus source-schema metadata for reporting."""

    matches: pd.DataFrame
    source_columns: tuple[str, ...]
    teams: tuple[str, ...]
    missing_values: dict[str, int]


def _bad_row_numbers(mask: pd.Series) -> list[int]:
    """Convert a Boolean mask to human-friendly CSV row numbers."""
    return [int(index) + 2 for index in mask.index[mask]]


def validate_season(frame: pd.DataFrame, spec: DownloadSpec) -> ValidatedSeason:
    """Validate a raw season and return the small canonical match table."""
    problems: list[str] = []
    missing_columns = sorted(REQUIRED_SOURCE_COLUMNS - set(frame.columns))
    if missing_columns:
        raise DataValidationError(
            spec.season, [f"missing required columns: {missing_columns}"]
        )

    required = frame[list(sorted(REQUIRED_SOURCE_COLUMNS))]
    null_counts = required.isna().sum()
    null_columns = {
        column: int(count) for column, count in null_counts.items() if count > 0
    }
    if null_columns:
        problems.append(f"null required values: {null_columns}")

    dates = pd.to_datetime(
        frame["Date"], dayfirst=True, errors="coerce", format="mixed"
    )
    invalid_dates = dates.isna()
    if invalid_dates.any():
        problems.append(
            f"invalid dates at CSV rows {_bad_row_numbers(invalid_dates)}"
        )

    for column in ("HomeTeam", "AwayTeam"):
        values = frame[column].astype("string")
        blank = values.isna() | values.str.strip().eq("")
        if blank.any():
            problems.append(
                f"blank {column} values at CSV rows {_bad_row_numbers(blank)}"
            )
        padded = values.notna() & values.ne(values.str.strip())
        if padded.any():
            problems.append(
                f"{column} values have surrounding whitespace at CSV rows "
                f"{_bad_row_numbers(padded)}"
            )

    same_team = frame["HomeTeam"].astype("string").str.strip().eq(
        frame["AwayTeam"].astype("string").str.strip()
    )
    if same_team.any():
        problems.append(
            f"home and away teams are identical at CSV rows "
            f"{_bad_row_numbers(same_team)}"
        )

    parsed_goals: dict[str, pd.Series] = {}
    for column in ("FTHG", "FTAG"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid = numeric.isna() | numeric.lt(0) | numeric.mod(1).ne(0)
        if invalid.any():
            problems.append(
                f"{column} must be a nonnegative integer at CSV rows "
                f"{_bad_row_numbers(invalid)}"
            )
        parsed_goals[column] = numeric

    outcomes = frame["FTR"].astype("string").str.strip()
    invalid_outcomes = ~outcomes.isin(RESULT_LABELS)
    if invalid_outcomes.any():
        problems.append(
            f"invalid FTR values at CSV rows {_bad_row_numbers(invalid_outcomes)}"
        )

    comparable = (
        parsed_goals["FTHG"].notna()
        & parsed_goals["FTAG"].notna()
        & outcomes.isin(RESULT_LABELS)
    )
    expected_outcome = pd.Series("D", index=frame.index, dtype="string")
    expected_outcome.loc[
        parsed_goals["FTHG"] > parsed_goals["FTAG"]
    ] = "H"
    expected_outcome.loc[
        parsed_goals["FTHG"] < parsed_goals["FTAG"]
    ] = "A"
    inconsistent = comparable & outcomes.ne(expected_outcome)
    if inconsistent.any():
        problems.append(
            f"FTR disagrees with full-time goals at CSV rows "
            f"{_bad_row_numbers(inconsistent)}"
        )

    valid_dates = dates.notna()
    outside = valid_dates & (
        dates.dt.date.lt(spec.date_min) | dates.dt.date.gt(spec.date_max)
    )
    if outside.any():
        problems.append(
            f"dates outside {spec.date_min}..{spec.date_max} at CSV rows "
            f"{_bad_row_numbers(outside)}"
        )

    fixture_key = pd.DataFrame(
        {
            "Date": dates,
            "HomeTeam": frame["HomeTeam"].astype("string").str.strip(),
            "AwayTeam": frame["AwayTeam"].astype("string").str.strip(),
        }
    )
    duplicate = fixture_key.duplicated(keep=False)
    if duplicate.any():
        problems.append(
            f"duplicate fixtures at CSV rows {_bad_row_numbers(duplicate)}"
        )

    teams = sorted(
        set(frame["HomeTeam"].dropna().astype(str).str.strip())
        | set(frame["AwayTeam"].dropna().astype(str).str.strip())
    )
    normalized: dict[str, set[str]] = {}
    for team in teams:
        key = "".join(character for character in team.casefold() if character.isalnum())
        normalized.setdefault(key, set()).add(team)
    collisions = [sorted(names) for names in normalized.values() if len(names) > 1]
    if collisions:
        problems.append(f"inconsistent team-name variants: {collisions}")

    if len(frame) != spec.expected_rows:
        problems.append(
            f"expected {spec.expected_rows} rows, found {len(frame)}"
        )
    if len(teams) != spec.expected_teams:
        problems.append(
            f"expected {spec.expected_teams} teams, found {len(teams)}"
        )

    if problems:
        raise DataValidationError(spec.season, problems)

    canonical = pd.DataFrame(
        {
            "season": spec.season,
            "match_date": dates.dt.strftime("%Y-%m-%d"),
            "home_team": frame["HomeTeam"].astype(str).str.strip(),
            "away_team": frame["AwayTeam"].astype(str).str.strip(),
            "full_time_home_goals": parsed_goals["FTHG"].astype("int64"),
            "full_time_away_goals": parsed_goals["FTAG"].astype("int64"),
            "result": outcomes.map(RESULT_LABELS),
        }
    )
    canonical = canonical.sort_values(
        ["match_date", "home_team", "away_team"], ignore_index=True
    )
    return ValidatedSeason(
        matches=canonical,
        source_columns=tuple(str(column) for column in frame.columns),
        teams=tuple(teams),
        missing_values={
            str(column): int(count)
            for column, count in frame.isna().sum().items()
            if count > 0
        },
    )
