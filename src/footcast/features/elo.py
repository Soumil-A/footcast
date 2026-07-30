"""Pre-match Elo ratings updated only after each completed result."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EloConfig:
    """Documented Elo constants for reproducible ratings."""

    initial_rating: float = 1500.0
    k_factor: float = 20.0
    home_advantage: float = 65.0
    rating_scale: float = 400.0


def expected_home_score(
    home_rating: float, away_rating: float, config: EloConfig
) -> float:
    """Return the expected home score after the fixed home adjustment."""
    adjusted_difference = (
        home_rating + config.home_advantage - away_rating
    )
    return 1.0 / (1.0 + 10.0 ** (-adjusted_difference / config.rating_scale))


@dataclass
class EloRatings:
    """Team ratings whose snapshots precede result-driven updates."""

    config: EloConfig = field(default_factory=EloConfig)
    ratings: dict[str, float] = field(default_factory=dict)

    def get(self, team: str) -> float:
        """Return a team's current pre-match rating."""
        return self.ratings.get(team, self.config.initial_rating)

    def update(self, home_team: str, away_team: str, result: str) -> None:
        """Update both ratings after a completed home/draw/away result."""
        actual_home = {
            "home_win": 1.0,
            "draw": 0.5,
            "away_win": 0.0,
        }.get(result)
        if actual_home is None:
            raise ValueError(f"Unsupported result for Elo update: {result}")

        home_before = self.get(home_team)
        away_before = self.get(away_team)
        expected_home = expected_home_score(
            home_before, away_before, self.config
        )
        change = self.config.k_factor * (actual_home - expected_home)
        self.ratings[home_team] = home_before + change
        self.ratings[away_team] = away_before - change
