"""Offline API and session tests for Phase 9 assistant chat."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from footcast.api.chat import (
    ChatService,
    ChatSessionStore,
    FixedWindowRateLimiter,
    assistant_from_environment,
)
from footcast.api.main import create_app
from footcast.assistant.client import (
    AssistantClientError,
    AssistantEvidence,
    AssistantRun,
)
from footcast.assistant.policy import ASSISTANT_POLICY_VERSION
from footcast.inference.elo_service import EloReferenceService


def prediction_service() -> EloReferenceService:
    matches = pd.DataFrame(
        [
            {
                "season": "2023-24",
                "split": "validation",
                "match_date": "2024-01-01",
                "home_team": "Alpha",
                "away_team": "Beta",
                "result": "home_win",
            }
        ]
    )
    return EloReferenceService(matches)


def assistant_run(answer: str = "Approved answer") -> AssistantRun:
    return AssistantRun(
        answer=answer,
        model="test-model",
        provider_responses=2,
        tool_calls=1,
        tool_names=("get_metric_definition",),
        evidence=(
            AssistantEvidence(
                tool_name="get_metric_definition",
                answer_mode="explanation",
                generated_at="2026-08-01T12:00:00Z",
                source="FootCast approved documentation",
                documentation_version="test-v1",
            ),
        ),
        input_tokens=20,
        output_tokens=5,
        total_tokens=25,
        estimated_cost_usd=None,
        latency_ms=42,
    )


class FakeAssistant:
    model = "test-model"

    def __init__(self, script: Sequence[AssistantRun | Exception] = ()) -> None:
        self.script = list(script)
        self.requests: list[tuple[str, list[dict[str, Any]]]] = []

    def answer(
        self,
        question: str,
        *,
        context: Sequence[Mapping[str, Any]] = (),
    ) -> AssistantRun:
        self.requests.append((question, [dict(item) for item in context]))
        result = self.script.pop(0) if self.script else assistant_run()
        if isinstance(result, Exception):
            raise result
        return result


def test_status_is_available_without_exposing_provider_configuration() -> None:
    app = create_app(prediction_service(), assistant_client=FakeAssistant())
    with TestClient(app) as client:
        response = client.get("/assistant/status")

    assert response.status_code == 200
    assert response.json() == {
        "available": True,
        "policy_version": ASSISTANT_POLICY_VERSION,
        "history_turn_limit": 10,
        "message_character_limit": 1000,
        "suggested_questions": [
            "What is Elo rating in simple terms?",
            "How have Arsenal performed over their last five completed matches?",
            "Compare Liverpool and Manchester City over their last five matches.",
            "What are FootCast's main model limitations?",
        ],
    }
    assert "model" not in response.json()


def test_unconfigured_status_stays_healthy_and_chat_returns_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FOOTCAST_LLM_MODEL", raising=False)
    with TestClient(create_app(prediction_service())) as client:
        status = client.get("/assistant/status")
        health = client.get("/health")
        chat = client.post("/assistant/chat", json={"message": "Hello"})

    assert status.json()["available"] is False
    assert health.status_code == 200
    assert chat.status_code == 503
    assert chat.json()["detail"] == "Assistant is not configured"


def test_chat_returns_session_answer_and_structured_evidence() -> None:
    assistant = FakeAssistant([assistant_run("Log loss measures probabilities.")])
    app = create_app(prediction_service(), assistant_client=assistant)
    with TestClient(app) as client:
        response = client.post(
            "/assistant/chat", json={"message": "What is log loss?"}
        )

    assert response.status_code == 200
    payload = response.json()
    UUID(payload["session_id"])
    assert payload["answer"] == "Log loss measures probabilities."
    assert payload["model"] == "test-model"
    assert payload["tool_calls"] == 1
    assert payload["latency_ms"] == 42
    assert payload["evidence"][0] == {
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
    assert "input_tokens" not in payload
    assert "estimated_cost_usd" not in payload


def test_follow_up_receives_only_prior_user_and_assistant_messages() -> None:
    assistant = FakeAssistant(
        [assistant_run("First answer"), assistant_run("Follow-up answer")]
    )
    app = create_app(prediction_service(), assistant_client=assistant)
    with TestClient(app) as client:
        first = client.post("/assistant/chat", json={"message": "First question"})
        second = client.post(
            "/assistant/chat",
            json={
                "message": "What about that?",
                "session_id": first.json()["session_id"],
            },
        )

    assert second.status_code == 200
    assert assistant.requests[0][1] == []
    assert assistant.requests[1][1] == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
    ]


def test_history_is_capped_at_ten_turns() -> None:
    assistant = FakeAssistant()
    service = ChatService(
        assistant,
        ip_limiter=FixedWindowRateLimiter(100),
        session_limiter=FixedWindowRateLimiter(100),
    )
    session_id = None
    for index in range(12):
        session_id, _ = service.chat(
            f"Question {index}",
            session_id=session_id,
            client_ip="127.0.0.1",
        )

    assert len(assistant.requests[-1][1]) == 20
    assert assistant.requests[-1][1][0]["content"] == "Question 1"
    assert assistant.requests[-1][1][-1]["content"] == "Approved answer"


def test_reset_clears_session_and_unknown_follow_up_is_rejected() -> None:
    with TestClient(
        create_app(prediction_service(), assistant_client=FakeAssistant())
    ) as client:
        first = client.post("/assistant/chat", json={"message": "Hello"})
        session_id = first.json()["session_id"]
        reset = client.delete(f"/assistant/sessions/{session_id}")
        repeat_reset = client.delete(f"/assistant/sessions/{session_id}")
        follow_up = client.post(
            "/assistant/chat",
            json={"message": "Again", "session_id": session_id},
        )

    assert reset.status_code == 200
    assert reset.json()["reset"] is True
    assert repeat_reset.status_code == 404
    assert repeat_reset.json()["reset"] is False
    assert follow_up.status_code == 404
    assert follow_up.json()["detail"] == "Unknown or expired assistant session"


def test_ip_rate_limit_returns_retry_after() -> None:
    service = ChatService(
        FakeAssistant(),
        ip_limiter=FixedWindowRateLimiter(1, clock=lambda: 10.0),
        session_limiter=FixedWindowRateLimiter(100),
    )
    with TestClient(create_app(prediction_service(), chat_service=service)) as client:
        first = client.post("/assistant/chat", json={"message": "One"})
        limited = client.post("/assistant/chat", json={"message": "Two"})

    assert first.status_code == 200
    assert limited.status_code == 429
    assert limited.headers["Retry-After"] == "60"
    assert limited.json()["detail"] == "Assistant rate limit exceeded"


def test_session_rate_limit_cannot_be_evaded_by_reusing_one_session() -> None:
    service = ChatService(
        FakeAssistant(),
        ip_limiter=FixedWindowRateLimiter(100),
        session_limiter=FixedWindowRateLimiter(1, clock=lambda: 10.0),
    )
    with TestClient(create_app(prediction_service(), chat_service=service)) as client:
        first = client.post("/assistant/chat", json={"message": "One"})
        session_id = first.json()["session_id"]
        limited = client.post(
            "/assistant/chat",
            json={"message": "Two", "session_id": session_id},
        )

    assert first.status_code == 200
    assert limited.status_code == 429


@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},
        {"message": "x" * 1_001},
        {"message": "hello", "unexpected": "field"},
        {"message": "hello", "session_id": "not-a-uuid"},
    ],
)
def test_chat_request_schema_is_strict(payload: dict[str, Any]) -> None:
    with TestClient(
        create_app(prediction_service(), assistant_client=FakeAssistant())
    ) as client:
        response = client.post("/assistant/chat", json=payload)
    assert response.status_code == 422


def test_request_body_limit_rejects_oversized_json_before_validation() -> None:
    with TestClient(
        create_app(prediction_service(), assistant_client=FakeAssistant())
    ) as client:
        response = client.post(
            "/assistant/chat",
            json={"message": "hello", "padding": "x" * 9_000},
        )
    assert response.status_code == 413
    assert response.json()["detail"] == "Assistant request body is too large"


def test_client_error_is_redacted_at_http_boundary() -> None:
    assistant = FakeAssistant([AssistantClientError("private provider detail")])
    app = create_app(prediction_service(), assistant_client=assistant)
    with TestClient(app) as client:
        response = client.post("/assistant/chat", json={"message": "Hello"})
    assert response.status_code == 502
    assert response.json()["detail"] == "Assistant could not complete the request"
    assert "private" not in response.text


def test_failed_new_request_does_not_leave_a_session() -> None:
    assistant = FakeAssistant(
        [AssistantClientError("failure"), assistant_run("success")]
    )
    sessions = ChatSessionStore()
    service = ChatService(assistant, sessions=sessions)
    with pytest.raises(AssistantClientError):
        service.chat("Fail", session_id=None, client_ip="127.0.0.1")

    new_id, result = service.chat(
        "Retry", session_id=None, client_ip="127.0.0.1"
    )
    assert isinstance(new_id, UUID)
    assert result.answer == "success"


def test_session_store_expires_context() -> None:
    now = [0.0]
    sessions = ChatSessionStore(ttl_seconds=5, clock=lambda: now[0])
    session_id, _ = sessions.open(None)
    sessions.append(session_id, "question", "answer")
    now[0] = 5.0
    with pytest.raises(KeyError, match="expired"):
        sessions.open(session_id)


def test_session_store_evicts_oldest_at_capacity() -> None:
    now = [0.0]
    sessions = ChatSessionStore(max_sessions=1, clock=lambda: now[0])
    oldest, _ = sessions.open(None)
    now[0] = 1.0
    newest, _ = sessions.open(None)

    assert newest != oldest
    with pytest.raises(KeyError, match="expired"):
        sessions.open(oldest)


def test_rate_limiter_resets_after_window() -> None:
    now = [0.0]
    limiter = FixedWindowRateLimiter(1, window_seconds=5, clock=lambda: now[0])
    assert limiter.check("key") is None
    assert limiter.check("key") == 5
    now[0] = 5.0
    assert limiter.check("key") is None


def test_environment_builder_is_disabled_without_any_assistant_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FOOTCAST_LLM_MODEL", raising=False)
    assert assistant_from_environment(prediction_service(), None) is None


def test_partial_environment_configuration_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FOOTCAST_LLM_MODEL", "model")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="analytics service"):
        assistant_from_environment(prediction_service(), None)


def test_session_id_type_is_uuid() -> None:
    assert isinstance(uuid4(), UUID)
