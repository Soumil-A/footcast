"""Replay completed matches and serve immutable Elo reference predictions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from footcast.data.download import DEFAULT_RAW_DIR
from footcast.data.matches import load_match_statistics
from footcast.evaluation.metrics import CLASS_LABELS, validate_probabilities
from footcast.features.elo import EloConfig, EloRatings, expected_home_score

REFERENCE_MODEL_VERSION = "footcast-elo-v2-reference"
REFERENCE_SPLITS = frozenset({"train", "validation", "test"})
INTENDED_USE = "educational probability demonstration"
LIMITATIONS = (
    "Not intended for betting or financial decisions.",
    "Draw probability is a historical training rate, not matchup-specific.",
    "Ratings omit injuries, lineups, transfers, and tactical context.",
    "The service uses completed data only through its displayed cutoff.",
)
DEVELOPMENT_EVIDENCE = {
    "rolling_mean_log_loss": 0.976,
    "rolling_mean_macro_f1": 0.401,
    "rolling_mean_draw_recall": 0.0,
    "original_2024_25_test_log_loss": 0.993,
}
REQUIRED_COLUMNS = frozenset(
    {
        "split",
        "match_date",
        "home_team",
        "away_team",
        "result",
    }
)


class PredictionInputError(ValueError):
    """Raised when a fixture cannot be safely scored."""


def reference_specification() -> dict[str, Any]:
    """Return the static, versioned Phase 7 reference contract."""
    return {
        "model_version": REFERENCE_MODEL_VERSION,
        "model_type": "Elo reference probability model",
        "class_order": list(CLASS_LABELS),
        "approved_splits": sorted(REFERENCE_SPLITS),
        "holdout_seasons_used": [],
        "elo_config": asdict(EloConfig()),
        "draw_probability": "empirical rate across approved completed matches",
        "development_evidence": DEVELOPMENT_EVIDENCE,
        "intended_use": INTENDED_USE,
        "limitations": list(LIMITATIONS),
    }


def reference_specification_sha256() -> str:
    """Hash the canonical reference contract for API provenance."""
    encoded = json.dumps(
        reference_specification(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MatchPrediction:
    """Versioned, auditable three-way fixture prediction."""

    home_team: str
    away_team: str
    match_date: date
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    predicted_result: str
    home_elo: float
    away_elo: float
    model_version: str
    data_cutoff: date
    intended_use: str
    warning: str

    def to_dict(self) -> dict[str, Any]:
        """Return a response-ready dictionary."""
        return asdict(self)


class EloReferenceService:
    """Immutable Elo state reconstructed from approved completed matches."""

    def __init__(
        self,
        matches: pd.DataFrame,
        *,
        elo_config: EloConfig | None = None,
    ) -> None:
        missing = sorted(REQUIRED_COLUMNS - set(matches.columns))
        if missing:
            raise ValueError(f"Elo service is missing columns: {missing}")
        if matches.empty:
            raise ValueError("Elo service requires completed matches")
        if "holdout" in set(matches["split"]):
            raise ValueError("Holdout data cannot initialize the prediction service")
        if not set(matches["split"]).issubset(REFERENCE_SPLITS):
            raise ValueError("Prediction service received an unsupported split")

        ordered = matches.copy()
        ordered["_parsed_date"] = pd.to_datetime(
            ordered["match_date"],
            errors="coerce",
        )
        if ordered["_parsed_date"].isna().any():
            raise ValueError("Every completed match date must be parseable")
        if ordered[list(REQUIRED_COLUMNS - {"match_date"})].isna().any().any():
            raise ValueError("Completed match fields cannot be missing")
        ordered = ordered.sort_values(
            ["_parsed_date", "home_team", "away_team"],
            ignore_index=True,
        )

        config = elo_config or EloConfig()
        elo = EloRatings(config)
        for match in ordered.to_dict(orient="records"):
            elo.update(
                str(match["home_team"]),
                str(match["away_team"]),
                str(match["result"]),
            )
        outcomes = ordered["result"].astype(str)
        unknown = set(outcomes) - set(CLASS_LABELS)
        if unknown:
            raise ValueError(f"Unsupported completed outcomes: {sorted(unknown)}")

        teams = sorted(
            set(ordered["home_team"].astype(str))
            | set(ordered["away_team"].astype(str))
        )
        self._config = config
        self._ratings = MappingProxyType(
            {team: float(elo.get(team)) for team in teams}
        )
        self._teams = tuple(teams)
        self._draw_probability = float((outcomes == "draw").mean())
        self._data_cutoff = ordered["_parsed_date"].max().date()
        self._match_count = int(len(ordered))

    @property
    def teams(self) -> tuple[str, ...]:
        """Return supported teams in stable display order."""
        return self._teams

    @property
    def data_cutoff(self) -> date:
        """Return the newest completed match date used by the service."""
        return self._data_cutoff

    @property
    def draw_probability(self) -> float:
        """Return the historical draw prior fitted from completed matches."""
        return self._draw_probability

    def rating(self, team: str) -> float:
        """Return one known team's immutable current rating."""
        normalized = team.strip()
        if normalized not in self._ratings:
            raise PredictionInputError(f"Unknown team: {normalized or team!r}")
        return self._ratings[normalized]

    def predict(
        self,
        home_team: str,
        away_team: str,
        match_date: date,
    ) -> MatchPrediction:
        """Score a future fixture without mutating ratings or histories."""
        home = home_team.strip()
        away = away_team.strip()
        if not home or not away:
            raise PredictionInputError("Home and away teams cannot be blank")
        if home == away:
            raise PredictionInputError("Home and away teams must be different")
        if match_date <= self._data_cutoff:
            raise PredictionInputError(
                f"match_date must be after data cutoff {self._data_cutoff.isoformat()}"
            )
        home_rating = self.rating(home)
        away_rating = self.rating(away)
        expected_home = expected_home_score(
            home_rating,
            away_rating,
            self._config,
        )
        decisive = 1.0 - self._draw_probability
        probabilities = validate_probabilities(
            np.asarray(
                [
                    [
                        decisive * expected_home,
                        self._draw_probability,
                        decisive * (1.0 - expected_home),
                    ]
                ]
            ),
            row_count=1,
        )[0]
        predicted = CLASS_LABELS[int(np.argmax(probabilities))]
        return MatchPrediction(
            home_team=home,
            away_team=away,
            match_date=match_date,
            home_win_probability=float(probabilities[0]),
            draw_probability=float(probabilities[1]),
            away_win_probability=float(probabilities[2]),
            predicted_result=predicted,
            home_elo=home_rating,
            away_elo=away_rating,
            model_version=REFERENCE_MODEL_VERSION,
            data_cutoff=self._data_cutoff,
            intended_use=INTENDED_USE,
            warning=LIMITATIONS[0],
        )

    def model_info(self) -> dict[str, Any]:
        """Return the transparent reference-model and evidence contract."""
        return {
            "model_version": REFERENCE_MODEL_VERSION,
            "specification_sha256": reference_specification_sha256(),
            "model_type": "Elo reference probability model",
            "class_order": list(CLASS_LABELS),
            "data_cutoff": self._data_cutoff,
            "completed_matches": self._match_count,
            "supported_team_count": len(self._teams),
            "draw_probability": self._draw_probability,
            "elo_config": asdict(self._config),
            "development_evidence": DEVELOPMENT_EVIDENCE,
            "intended_use": INTENDED_USE,
            "limitations": list(LIMITATIONS),
            "holdout_seasons_used": [],
        }


def load_reference_service() -> EloReferenceService:
    """Build the local API service without reading the sealed holdout."""
    matches = load_match_statistics(
        DEFAULT_RAW_DIR,
        splits=REFERENCE_SPLITS,
    )
    return EloReferenceService(matches)
