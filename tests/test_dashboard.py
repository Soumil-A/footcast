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


def test_client_requests_portfolio_summary() -> None:
    captured = {}

    def opener(request, *, timeout):
        captured["url"] = request.full_url
        return _Response({"completed_matches": 3800})

    result = FootCastApiClient("http://localhost:8000", opener=opener).portfolio()

    assert result == {"completed_matches": 3800}
    assert captured["url"] == "http://localhost:8000/analytics/portfolio"


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


def test_client_uses_server_side_assistant_contract() -> None:
    captured = []

    def opener(request, *, timeout):
        captured.append(
            {
                "url": request.full_url,
                "method": request.method,
                "body": None if request.data is None else json.loads(request.data),
                "timeout": timeout,
            }
        )
        if request.method == "DELETE":
            return _Response({"reset": True})
        if request.full_url.endswith("/assistant/status"):
            return _Response({"available": True})
        return _Response({"session_id": "session-1", "answer": "Grounded"})

    client = FootCastApiClient("http://localhost:8000", opener=opener)

    assert client.assistant_status() == {"available": True}
    assert client.chat("First question")["answer"] == "Grounded"
    client.chat("Follow up", session_id="session-1")
    assert client.reset_assistant_session("session-1") == {"reset": True}

    assert captured == [
        {
            "url": "http://localhost:8000/assistant/status",
            "method": "GET",
            "body": None,
            "timeout": 10.0,
        },
        {
            "url": "http://localhost:8000/assistant/chat",
            "method": "POST",
            "body": {"message": "First question"},
            "timeout": 35.0,
        },
        {
            "url": "http://localhost:8000/assistant/chat",
            "method": "POST",
            "body": {"message": "Follow up", "session_id": "session-1"},
            "timeout": 35.0,
        },
        {
            "url": "http://localhost:8000/assistant/sessions/session-1",
            "method": "DELETE",
            "body": None,
            "timeout": 10.0,
        },
    ]


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
    def __init__(self) -> None:
        self.chat_requests: list[dict] = []
        self.reset_requests: list[str] = []

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

    def portfolio(self) -> dict:
        return {
            "completed_matches": 3800,
            "first_match_date": "2015-08-08",
            "data_cutoff": "2025-05-25",
            "season_count": 10,
            "outcome_distribution": [
                {"outcome": "home_win", "matches": 1660, "share": 0.437},
                {"outcome": "draw", "matches": 884, "share": 0.233},
                {"outcome": "away_win", "matches": 1256, "share": 0.330},
            ],
            "strength_ranking": [
                {"rank": 1, "team": "Arsenal", "elo": 1600.0},
                {"rank": 2, "team": "Chelsea", "elo": 1500.0},
            ],
            "test_season": "2024-25",
            "test_matches": 380,
            "benchmarks": [
                {
                    "model": "Elo (deployed)",
                    "accuracy": 0.526,
                    "macro_f1": 0.392,
                    "log_loss": 0.993,
                },
                {
                    "model": "Frozen Random Forest",
                    "accuracy": 0.513,
                    "macro_f1": 0.383,
                    "log_loss": 1.005,
                },
            ],
            "deployed_elo_recall": {
                "home_win": 0.845,
                "draw": 0.0,
                "away_win": 0.523,
            },
            "deployed_elo_confusion_matrix": [
                [131, 0, 24],
                [63, 0, 30],
                [63, 0, 69],
            ],
            "class_order": ["home_win", "draw", "away_win"],
            "selection_note": "Elo is the transparent reference model.",
        }

    def assistant_status(self) -> dict:
        return {
            "available": True,
            "message_character_limit": 1_000,
            "suggested_questions": [
                "What is Elo rating in simple terms?",
                "How have Arsenal performed over their last five completed matches?",
                "Compare Liverpool and Manchester City over their last five matches.",
                "What are FootCast's main model limitations?",
            ],
        }

    def chat(self, message: str, *, session_id: str | None = None) -> dict:
        self.chat_requests.append({"message": message, "session_id": session_id})
        return {
            "session_id": "00000000-0000-0000-0000-000000000001",
            "answer": "Elo is a relative strength rating based on past results.",
            "model": "test-language-model",
            "evidence": [
                {
                    "tool_name": "get_metric_definition",
                    "answer_mode": "explanation",
                    "generated_at": "2026-08-01T12:00:00Z",
                    "source": "FootCast approved documentation",
                    "data_cutoff": None,
                    "model_version": None,
                    "test_season": None,
                    "window": None,
                    "sample_size": None,
                    "documentation_version": "test-v1",
                }
            ],
            "tool_calls": 1,
            "latency_ms": 125,
        }

    def reset_assistant_session(self, session_id: str) -> dict:
        self.reset_requests.append(session_id)
        return {"session_id": session_id, "reset": True}

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
    client = _DashboardClient()
    app = AppTest.from_function(
        _render_test_dashboard, args=(client,)
    ).run(timeout=10)

    assert not app.exception
    assert len(app.selectbox) == 2
    assert [tab.label for tab in app.tabs] == [
        "01 · Match Forecast",
        "02 · Team Analytics",
        "03 · Model Insights",
        "04 · Ask FootCast",
    ]
    assert any("FootCast" in element.value for element in app.markdown)
    assert any(
        "Launch sequence ready" in element.value
        for element in app.markdown
    )
    rendered_markup = "\n".join(element.value for element in app.markdown)
    assert "Approved history" in rendered_markup
    assert "3,800" in rendered_markup
    assert "Known limitation" in rendered_markup
    assert "Future fixture" in rendered_markup
    assert "Data link verified" in rendered_markup

    forecast_button = next(
        button for button in app.button if button.label == "Generate forecast →"
    )
    forecast_button.click().run(timeout=10)

    assert not app.exception
    assert any("54.0%" in element.value for element in app.markdown)
    assert any("Highest model probability" in element.value for element in app.markdown)


def test_streamlit_chat_renders_grounded_answer_and_evidence() -> None:
    client = _DashboardClient()
    app = AppTest.from_function(
        _render_test_dashboard, args=(client,)
    ).run(timeout=10)

    assert len(app.chat_input) == 1
    app.chat_input[0].set_value("Explain Elo").run(timeout=10)

    assert not app.exception
    assert client.chat_requests == [{"message": "Explain Elo", "session_id": None}]
    assert any(
        "Approved explanation" in element.value for element in app.markdown
    )
    assert any(
        "get_metric_definition" in element.value for element in app.markdown
    )
    assert any(
        "FootCast approved documentation" in element.value
        for element in app.markdown
    )
    assert any(
        "Language layer: test-language-model" in element.value
        for element in app.caption
    )


def test_streamlit_suggested_question_and_reset_flow() -> None:
    client = _DashboardClient()
    app = AppTest.from_function(
        _render_test_dashboard, args=(client,)
    ).run(timeout=10)

    suggestion = next(
        button
        for button in app.button
        if button.label == "What is Elo rating in simple terms?"
    )
    suggestion.click().run(timeout=10)
    assert client.chat_requests[0]["message"] == suggestion.label

    reset = next(button for button in app.button if button.label == "Reset chat")
    reset.click().run(timeout=10)

    assert client.reset_requests == [
        "00000000-0000-0000-0000-000000000001"
    ]
    assert not app.exception


class _UnavailableAssistantClient(_DashboardClient):
    def assistant_status(self) -> dict:
        return {
            "available": False,
            "message_character_limit": 1_000,
            "suggested_questions": [],
        }

    def chat(self, message: str, *, session_id: str | None = None) -> dict:
        raise AssertionError("Unavailable assistant must not be called")


def test_streamlit_chat_has_safe_unavailable_state() -> None:
    app = AppTest.from_function(
        _render_test_dashboard, args=(_UnavailableAssistantClient(),)
    ).run(timeout=10)

    assert not app.exception
    assert len(app.chat_input) == 0
    assert any(
        "Conversation layer is safely offline" in element.value
        for element in app.markdown
    )
