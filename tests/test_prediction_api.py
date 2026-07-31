"""Tests for the Phase 7 Elo service and FastAPI contract."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from footcast.api.main import create_app
from footcast.features.elo import EloConfig
from footcast.inference.elo_service import (
    REFERENCE_MODEL_VERSION,
    REFERENCE_SPLITS,
    EloReferenceService,
    PredictionInputError,
    reference_specification,
    reference_specification_sha256,
)


def _completed_matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": "2023-24",
                "split": "validation",
                "match_date": "2024-01-01",
                "home_team": "Alpha",
                "away_team": "Beta",
                "result": "home_win",
            },
            {
                "season": "2023-24",
                "split": "validation",
                "match_date": "2024-01-08",
                "home_team": "Gamma",
                "away_team": "Alpha",
                "result": "away_win",
            },
            {
                "season": "2023-24",
                "split": "validation",
                "match_date": "2024-01-15",
                "home_team": "Beta",
                "away_team": "Gamma",
                "result": "draw",
            },
        ]
    )


@pytest.fixture
def service() -> EloReferenceService:
    return EloReferenceService(
        _completed_matches(),
        elo_config=EloConfig(home_advantage=0.0),
    )


def test_service_replays_completed_matches_and_exposes_cutoff(
    service: EloReferenceService,
) -> None:
    assert service.teams == ("Alpha", "Beta", "Gamma")
    assert service.data_cutoff == date(2024, 1, 15)
    assert service.draw_probability == pytest.approx(1 / 3)
    assert service.rating("Alpha") > service.rating("Beta")


def test_prediction_is_valid_and_does_not_mutate_ratings(
    service: EloReferenceService,
) -> None:
    before = {team: service.rating(team) for team in service.teams}

    prediction = service.predict("Alpha", "Beta", date(2024, 2, 1))
    repeat = service.predict("Alpha", "Beta", date(2024, 2, 1))

    probabilities = [
        prediction.home_win_probability,
        prediction.draw_probability,
        prediction.away_win_probability,
    ]
    assert sum(probabilities) == pytest.approx(1.0)
    assert all(0.0 <= probability <= 1.0 for probability in probabilities)
    assert prediction.to_dict() == repeat.to_dict()
    assert {team: service.rating(team) for team in service.teams} == before
    assert prediction.model_version == REFERENCE_MODEL_VERSION
    assert prediction.data_cutoff == date(2024, 1, 15)


@pytest.mark.parametrize(
    ("home", "away", "match_date", "message"),
    [
        ("Alpha", "Alpha", date(2024, 2, 1), "must be different"),
        ("Unknown", "Beta", date(2024, 2, 1), "Unknown team"),
        ("Alpha", "Beta", date(2024, 1, 15), "must be after"),
    ],
)
def test_prediction_rejects_invalid_fixtures(
    service: EloReferenceService,
    home: str,
    away: str,
    match_date: date,
    message: str,
) -> None:
    with pytest.raises(PredictionInputError, match=message):
        service.predict(home, away, match_date)


def test_service_rejects_holdout_rows() -> None:
    matches = _completed_matches()
    matches.loc[0, "split"] = "holdout"

    with pytest.raises(ValueError, match="Holdout"):
        EloReferenceService(matches)


def test_reference_loader_contract_excludes_holdout() -> None:
    assert REFERENCE_SPLITS == {"train", "validation", "test"}
    assert "holdout" not in REFERENCE_SPLITS


def test_model_info_is_versioned_and_discloses_limitations(
    service: EloReferenceService,
) -> None:
    info = service.model_info()

    assert info["model_version"] == REFERENCE_MODEL_VERSION
    assert info["specification_sha256"] == reference_specification_sha256()
    assert info["class_order"] == ["home_win", "draw", "away_win"]
    assert info["holdout_seasons_used"] == []
    assert info["limitations"]
    assert info["supported_team_count"] == 3


def test_tracked_reference_spec_matches_code_contract() -> None:
    path = Path(__file__).parents[1] / "models" / "elo_reference_spec.json"
    tracked = json.loads(path.read_text(encoding="utf-8"))

    assert tracked == reference_specification()


def test_api_health_teams_prediction_and_model_info(
    service: EloReferenceService,
) -> None:
    with TestClient(create_app(service)) as client:
        health = client.get("/health")
        teams = client.get("/teams")
        prediction = client.post(
            "/predict",
            json={
                "home_team": "Alpha",
                "away_team": "Beta",
                "match_date": "2024-02-01",
            },
        )
        info = client.get("/model/info")

    assert health.status_code == 200
    assert health.json()["holdout_used"] is False
    assert float(health.headers["X-Process-Time-Ms"]) >= 0.0
    assert teams.json()["teams"] == ["Alpha", "Beta", "Gamma"]
    assert prediction.status_code == 200
    assert prediction.json()["model_version"] == REFERENCE_MODEL_VERSION
    assert info.json()["holdout_seasons_used"] == []


def test_api_returns_structured_validation_errors(
    service: EloReferenceService,
) -> None:
    with TestClient(create_app(service)) as client:
        same_team = client.post(
            "/predict",
            json={
                "home_team": "Alpha",
                "away_team": "Alpha",
                "match_date": "2024-02-01",
            },
        )
        leaked_field = client.post(
            "/predict",
            json={
                "home_team": "Alpha",
                "away_team": "Beta",
                "match_date": "2024-02-01",
                "result": "home_win",
            },
        )

    assert same_team.status_code == 422
    assert same_team.json()["detail"] == "Home and away teams must be different"
    assert leaked_field.status_code == 422
    assert leaked_field.json()["detail"][0]["type"] == "extra_forbidden"
