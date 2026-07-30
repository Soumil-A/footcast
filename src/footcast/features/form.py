"""Stateful team histories that expose only information from completed matches."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

ROLLING_WINDOW = 5


@dataclass(frozen=True)
class CompletedTeamMatch:
    """One team perspective after a match has finished."""

    match_date: date
    season: str
    location: str
    points: int
    win: int
    goals_for: int
    goals_against: int
    shots: float
    shots_on_target: float


@dataclass
class TeamHistory:
    """Mutable completed-match state for one team."""

    completed: list[CompletedTeamMatch] = field(default_factory=list)
    last_match_date: date | None = None
    total_goals_for: int = 0
    total_goals_against: int = 0
    current_season: str | None = None
    season_matches: int = 0
    season_points: int = 0

    def _ensure_season(self, season: str) -> None:
        if self.current_season != season:
            self.current_season = season
            self.season_matches = 0
            self.season_points = 0

    def snapshot(
        self, *, match_date: date, season: str, location: str
    ) -> dict[str, float | int | None]:
        """Return pre-match values without mutating completed history."""
        self._ensure_season(season)
        recent = self.completed[-ROLLING_WINDOW:]
        venue_recent = [
            match
            for match in self.completed
            if match.location == location
        ][-ROLLING_WINDOW:]
        history_matches = len(self.completed)
        days_since = (
            (match_date - self.last_match_date).days
            if self.last_match_date is not None
            else None
        )
        if days_since is not None and days_since < 0:
            raise ValueError("Matches must be processed chronologically per team")

        return {
            "history_matches": history_matches,
            "rolling_matches": len(recent),
            "form_points_last_5": sum(match.points for match in recent),
            "form_wins_last_5": sum(match.win for match in recent),
            "goals_for_last_5": sum(match.goals_for for match in recent),
            "goals_against_last_5": sum(
                match.goals_against for match in recent
            ),
            "shots_last_5": sum(match.shots for match in recent),
            "shots_on_target_last_5": sum(
                match.shots_on_target for match in recent
            ),
            "venue_matches_last_5": len(venue_recent),
            "venue_points_last_5": sum(
                match.points for match in venue_recent
            ),
            "days_since_previous_match": days_since,
            "season_matches_played": self.season_matches,
            "season_points": self.season_points,
            "expanding_goals_for_mean": (
                self.total_goals_for / history_matches
                if history_matches
                else 0.0
            ),
            "expanding_goals_against_mean": (
                self.total_goals_against / history_matches
                if history_matches
                else 0.0
            ),
        }

    def record(self, match: CompletedTeamMatch) -> None:
        """Add a match only after its pre-match snapshot has been captured."""
        self._ensure_season(match.season)
        if self.last_match_date is not None and match.match_date < self.last_match_date:
            raise ValueError("Matches must be recorded chronologically per team")
        self.completed.append(match)
        self.last_match_date = match.match_date
        self.total_goals_for += match.goals_for
        self.total_goals_against += match.goals_against
        self.season_matches += 1
        self.season_points += match.points
