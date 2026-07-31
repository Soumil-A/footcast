"""Tests for the dashboard's HTTP boundary."""

from __future__ import annotations

import io
import json
from urllib.error import HTTPError, URLError

import pytest
from streamlit.testing.v1 import AppTest

from footcast.dashboard.client import FootCastApiClient, FootCastApiError


class _Response:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_client_builds_encoded_analytics_request() -> None:
    captured = {}

    def opener(request, *, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return _Response({"matches": []})

    client = FootCastApiClient("http://localhost:8000/", opener=opener)
    result = client.head_to_head("Team A", "Team & B", limit=7)

    assert result == {"matches": []}
    assert captured["url"] == (
        "http://localhost:8000/analytics/head-to-head?"
        "team_a=Team+A&team_b=Team+%26+B&limit=7"
    )
    assert captured["timeout"] == 10.0


def test_client_posts_only_pre_match_fixture_fields() -> None:
    captured = {}

    def opener(request, *, timeout):
        captured["method"] = request.method
        captured["body"] = json.loads(request.data)
        return _Response({"predicted_result": "home_win"})

    client = FootCastApiClient("http://localhost:8000", opener=opener)
    response = client.predict("Alpha", "Beta", "2026-08-01")

    assert response["predicted_result"] == "home_win"
    assert captured == {
        "method": "POST",
        "body": {
            "home_team": "Alpha",
            "away_team": "Beta",
            "match_date": "2026-08-01",
        },
    }


def test_client_surfaces_api_validation_detail() -> None:
    error = HTTPError(
        "http://localhost/predict",
        422,
        "Unprocessable Entity",
        {},
        io.BytesIO(b'{"detail":"Teams must be different"}'),
    )

    def opener(_request, *, timeout):
        raise error

    with pytest.raises(FootCastApiError, match="Teams must be different"):
        FootCastApiClient("http://localhost", opener=opener).teams()


def test_client_surfaces_unavailable_api() -> None:
    def opener(_request, *, timeout):
        raise URLError("connection refused")

    with pytest.raises(FootCastApiError, match="unavailable"):
        FootCastApiClient("http://localhost", opener=opener).health()


class _DashboardClient:
    def teams(self) -> dict:
        return {"teams": ["Arsenal", "Chelsea"]}

    def model_info(self) -> dict:
        return {
            "model_version": "test-elo",
            "data_cutoff": "2025-05-25",
            "intended_use": "educational demonstration",
            "limitations": ["Not betting advice."],
            "specification_sha256": "abc123def456",
            "completed_matches": 3,
        }

    def compare(self, _home: str, _away: str, *, limit: int) -> dict:
        form = {
            "data_cutoff": "2025-05-25",
            "summary": {
                "matches": 1,
                "wins": 1,
                "draws": 0,
                "losses": 0,
                "points": 3,
                "goals_for": 2,
                "goals_against": 1,
            },
            "matches": [
                {
                    "match_date": "2025-05-25",
                    "opponent": "Chelsea",
                    "venue": "home",
                    "goals_for": 2,
                    "goals_against": 1,
                    "outcome": "win",
                    "points": 3,
                }
            ],
        }
        return {
            "home": {"team": "Arsenal", **form},
            "away": {"team": "Chelsea", **form},
            "home_elo": 1600.0,
            "away_elo": 1500.0,
            "elo_difference": 100.0,
            "data_cutoff": "2025-05-25",
        }

    def head_to_head(self, _home: str, _away: str, *, limit: int) -> dict:
        return {"matches": []}

    def predict(self, home: str, away: str, match_date: str) -> dict:
        return {
            "home_team": home,
            "away_team": away,
            "match_date": match_date,
            "home_win_probability": 0.54,
            "draw_probability": 0.23,
            "away_win_probability": 0.23,
            "predicted_result": "home_win",
        }


def _render_test_dashboard(client) -> None:
    from footcast.dashboard.app import render_dashboard

    render_dashboard(client)


def test_streamlit_dashboard_renders_against_client_contract() -> None:
    app = AppTest.from_function(
        _render_test_dashboard, args=(_DashboardClient(),)
    ).run(timeout=10)

    assert not app.exception
    assert len(app.selectbox) == 2
    assert any("FootCast" in element.value for element in app.markdown)
    assert any(
        "Forecast engine standing by" in element.value
        for element in app.markdown
    )

    assert len(app.button) == 1
    app.button[0].click().run(timeout=10)

    assert not app.exception
    assert any("54.0%" in element.value for element in app.markdown)
    assert any("Highest model probability" in element.value for element in app.markdown)
