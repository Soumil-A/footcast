"""Read-only recent-form and head-to-head analytics."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from footcast.inference.elo_service import REFERENCE_SPLITS

REQUIRED_ANALYTICS_COLUMNS = frozenset(
    {
        "split",
        "match_date",
        "home_team",
        "away_team",
        "full_time_home_goals",
        "full_time_away_goals",
        "result",
    }
)
RESULTS = frozenset({"home_win", "draw", "away_win"})


class AnalyticsInputError(ValueError):
    """Raised when an analytics request cannot be answered safely."""


class AnalyticsService:
    """Immutable views over approved completed fixtures."""

    def __init__(self, matches: pd.DataFrame) -> None:
        missing = sorted(REQUIRED_ANALYTICS_COLUMNS - set(matches.columns))
        if missing:
            raise ValueError(f"Analytics service is missing columns: {missing}")
        if matches.empty:
            raise ValueError("Analytics service requires completed matches")
        splits = set(matches["split"].astype(str))
        if "holdout" in splits:
            raise ValueError("Holdout data cannot initialize analytics")
        if not splits.issubset(REFERENCE_SPLITS):
            raise ValueError("Analytics service received an unsupported split")

        history = matches.loc[:, sorted(REQUIRED_ANALYTICS_COLUMNS)].copy()
        history["match_date"] = pd.to_datetime(
            history["match_date"], errors="coerce"
        )
        if history.isna().any().any():
            raise ValueError("Completed analytics fields cannot be missing")
        if not set(history["result"].astype(str)).issubset(RESULTS):
            raise ValueError("Analytics service received an unsupported result")

        for column in ("full_time_home_goals", "full_time_away_goals"):
            numeric = pd.to_numeric(history[column], errors="coerce")
            if numeric.isna().any() or (numeric < 0).any() or (numeric % 1 != 0).any():
                raise ValueError("Completed match goals must be nonnegative integers")
            history[column] = numeric.astype(int)

        history = history.sort_values(
            ["match_date", "home_team", "away_team"], ignore_index=True
        )
        self._history = history
        self._teams = tuple(
            sorted(
                set(history["home_team"].astype(str))
                | set(history["away_team"].astype(str))
            )
        )
        self._data_cutoff = history["match_date"].max().date()

    @property
    def teams(self) -> tuple[str, ...]:
        return self._teams

    @property
    def data_cutoff(self) -> date:
        return self._data_cutoff

    def _team(self, team: str) -> str:
        normalized = team.strip()
        if normalized not in self._teams:
            raise AnalyticsInputError(f"Unknown team: {normalized or team!r}")
        return normalized

    @staticmethod
    def _limit(limit: int) -> int:
        if not 1 <= limit <= 20:
            raise AnalyticsInputError("limit must be between 1 and 20")
        return limit

    def recent_form(self, team: str, *, limit: int = 5) -> dict[str, Any]:
        """Return latest completed fixtures from one team's perspective."""
        selected_team = self._team(team)
        count = self._limit(limit)
        rows = self._history.loc[
            (self._history["home_team"] == selected_team)
            | (self._history["away_team"] == selected_team)
        ].tail(count)

        matches = [
            self._team_view(row, selected_team)
            for _, row in rows.iloc[::-1].iterrows()
        ]
        outcomes = [match["outcome"] for match in matches]
        return {
            "team": selected_team,
            "data_cutoff": self._data_cutoff,
            "summary": {
                "matches": len(matches),
                "wins": outcomes.count("win"),
                "draws": outcomes.count("draw"),
                "losses": outcomes.count("loss"),
                "points": sum(int(match["points"]) for match in matches),
                "goals_for": sum(int(match["goals_for"]) for match in matches),
                "goals_against": sum(
                    int(match["goals_against"]) for match in matches
                ),
            },
            "matches": matches,
        }

    def compare(
        self, home_team: str, away_team: str, *, limit: int = 5
    ) -> dict[str, Any]:
        """Return side-by-side recent form for two distinct known teams."""
        home = self._team(home_team)
        away = self._team(away_team)
        if home == away:
            raise AnalyticsInputError("Teams must be different")
        count = self._limit(limit)
        return {
            "home": self.recent_form(home, limit=count),
            "away": self.recent_form(away, limit=count),
            "data_cutoff": self._data_cutoff,
        }

    def head_to_head(
        self, team_a: str, team_b: str, *, limit: int = 10
    ) -> dict[str, Any]:
        """Return latest meetings, preserving the historical venue orientation."""
        first = self._team(team_a)
        second = self._team(team_b)
        if first == second:
            raise AnalyticsInputError("Teams must be different")
        count = self._limit(limit)
        first_home = (self._history["home_team"] == first) & (
            self._history["away_team"] == second
        )
        second_home = (self._history["home_team"] == second) & (
            self._history["away_team"] == first
        )
        rows = self._history.loc[first_home | second_home].tail(count)
        matches: list[dict[str, Any]] = []
        for _, row in rows.iloc[::-1].iterrows():
            view = self._team_view(row, first)
            matches.append(
                {
                    "match_date": view["match_date"],
                    "home_team": str(row["home_team"]),
                    "away_team": str(row["away_team"]),
                    "home_goals": int(row["full_time_home_goals"]),
                    "away_goals": int(row["full_time_away_goals"]),
                    "team_a_outcome": view["outcome"],
                }
            )
        return {
            "team_a": first,
            "team_b": second,
            "data_cutoff": self._data_cutoff,
            "matches": matches,
        }

    @staticmethod
    def _team_view(row: pd.Series, team: str) -> dict[str, Any]:
        is_home = str(row["home_team"]) == team
        goals_for = int(
            row["full_time_home_goals"] if is_home else row["full_time_away_goals"]
        )
        goals_against = int(
            row["full_time_away_goals"] if is_home else row["full_time_home_goals"]
        )
        if goals_for > goals_against:
            outcome, points = "win", 3
        elif goals_for == goals_against:
            outcome, points = "draw", 1
        else:
            outcome, points = "loss", 0
        return {
            "match_date": row["match_date"].date(),
            "opponent": str(row["away_team"] if is_home else row["home_team"]),
            "venue": "home" if is_home else "away",
            "goals_for": goals_for,
            "goals_against": goals_against,
            "outcome": outcome,
            "points": points,
        }
