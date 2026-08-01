"""Typed, rate-limited HTTP boundary for the FootCast assistant."""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Protocol
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from footcast.analytics.service import AnalyticsService
from footcast.assistant.client import (
    AssistantClient,
    AssistantClientError,
    AssistantEvidence,
    AssistantRun,
    ProviderUnavailableError,
)
from footcast.assistant.openai_provider import OpenAIResponsesProvider
from footcast.assistant.policy import ASSISTANT_POLICY_VERSION
from footcast.assistant.settings import AssistantSettings
from footcast.assistant.tools import AssistantTools
from footcast.inference.elo_service import EloReferenceService

LOGGER = logging.getLogger(__name__)
MAX_HISTORY_ITEMS = 20
SUGGESTED_QUESTIONS = (
    "What is Elo rating in simple terms?",
    "How have Arsenal performed over their last five completed matches?",
    "Compare Liverpool and Manchester City over their last five matches.",
    "What are FootCast's main model limitations?",
)


class StrictAPIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ChatRequest(StrictAPIModel):
    message: str = Field(min_length=1, max_length=1_000)
    session_id: UUID | None = None


class ChatEvidenceResponse(StrictAPIModel):
    tool_name: str
    answer_mode: str
    generated_at: datetime
    source: str
    data_cutoff: date | None = None
    model_version: str | None = None
    test_season: str | None = None
    window: int | None = None
    sample_size: dict[str, int] | None = None
    documentation_version: str | None = None


class ChatResponse(StrictAPIModel):
    session_id: UUID
    answer: str
    model: str
    evidence: list[ChatEvidenceResponse]
    tool_calls: int = Field(ge=0, le=4)
    latency_ms: int = Field(ge=0)


class ChatStatusResponse(StrictAPIModel):
    available: bool
    policy_version: str
    history_turn_limit: int
    message_character_limit: int
    suggested_questions: list[str]


class ResetResponse(StrictAPIModel):
    session_id: UUID
    reset: bool


class AssistantAnswerer(Protocol):
    @property
    def model(self) -> str: ...

    def answer(
        self,
        question: str,
        *,
        context: Sequence[Mapping[str, Any]] = (),
    ) -> AssistantRun: ...


@dataclass
class _RateBucket:
    window_started: float
    requests: int = 0


class FixedWindowRateLimiter:
    """Small in-memory limiter suitable for one free-tier API process."""

    def __init__(
        self,
        limit: int,
        *,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if limit < 1 or window_seconds <= 0:
            raise ValueError("Rate limit and window must be positive")
        self._limit = limit
        self._window_seconds = window_seconds
        self._clock = clock
        self._buckets: dict[str, _RateBucket] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> int | None:
        """Consume one request or return the number of seconds until reset."""
        now = self._clock()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None or now - bucket.window_started >= self._window_seconds:
                self._buckets[key] = _RateBucket(now, 1)
                self._remove_expired(now)
                return None
            if bucket.requests >= self._limit:
                return max(
                    1,
                    math.ceil(
                        self._window_seconds - (now - bucket.window_started)
                    ),
                )
            bucket.requests += 1
            return None

    def _remove_expired(self, now: float) -> None:
        expired = [
            key
            for key, bucket in self._buckets.items()
            if now - bucket.window_started >= self._window_seconds
        ]
        for key in expired:
            self._buckets.pop(key, None)


@dataclass
class _ChatSession:
    history: list[dict[str, str]] = field(default_factory=list)
    last_accessed: float = 0.0


class ChatSessionStore:
    """Ephemeral, bounded storage for minimum follow-up context."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 3_600,
        max_sessions: int = 1_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0 or max_sessions < 1:
            raise ValueError("Session TTL and capacity must be positive")
        self._ttl_seconds = ttl_seconds
        self._max_sessions = max_sessions
        self._clock = clock
        self._sessions: dict[UUID, _ChatSession] = {}
        self._lock = threading.Lock()

    def open(self, session_id: UUID | None) -> tuple[UUID, list[dict[str, str]]]:
        now = self._clock()
        with self._lock:
            self._remove_expired(now)
            if session_id is None:
                self._make_room()
                session_id = uuid4()
                self._sessions[session_id] = _ChatSession(last_accessed=now)
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError("Unknown or expired assistant session")
            session.last_accessed = now
            return session_id, [dict(item) for item in session.history]

    def append(self, session_id: UUID, question: str, answer: str) -> None:
        now = self._clock()
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError("Unknown or expired assistant session")
            session.history.extend(
                (
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                )
            )
            session.history = session.history[-MAX_HISTORY_ITEMS:]
            session.last_accessed = now

    def reset(self, session_id: UUID) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def _remove_expired(self, now: float) -> None:
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.last_accessed >= self._ttl_seconds
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)

    def _make_room(self) -> None:
        if len(self._sessions) < self._max_sessions:
            return
        oldest = min(
            self._sessions,
            key=lambda session_id: self._sessions[session_id].last_accessed,
        )
        self._sessions.pop(oldest)


class ChatService:
    """Coordinate availability, session context, and abuse controls."""

    def __init__(
        self,
        assistant: AssistantAnswerer | None,
        *,
        sessions: ChatSessionStore | None = None,
        ip_limiter: FixedWindowRateLimiter | None = None,
        session_limiter: FixedWindowRateLimiter | None = None,
    ) -> None:
        self._assistant = assistant
        self._sessions = sessions or ChatSessionStore()
        self._ip_limiter = ip_limiter or FixedWindowRateLimiter(20)
        self._session_limiter = session_limiter or FixedWindowRateLimiter(10)

    @property
    def available(self) -> bool:
        return self._assistant is not None

    def chat(
        self,
        message: str,
        *,
        session_id: UUID | None,
        client_ip: str,
    ) -> tuple[UUID, AssistantRun]:
        if self._assistant is None:
            raise ProviderUnavailableError("Assistant is not configured")
        self._enforce_limit(self._ip_limiter, f"ip:{client_ip}")
        created = session_id is None
        try:
            active_id, context = self._sessions.open(session_id)
            self._enforce_limit(
                self._session_limiter, f"session:{active_id}"
            )
            result = self._assistant.answer(message, context=context)
            self._sessions.append(active_id, message, result.answer)
            return active_id, result
        except Exception:
            if created and "active_id" in locals():
                self._sessions.reset(active_id)
            raise

    def reset(self, session_id: UUID) -> bool:
        return self._sessions.reset(session_id)

    @staticmethod
    def _enforce_limit(limiter: FixedWindowRateLimiter, key: str) -> None:
        retry_after = limiter.check(key)
        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail="Assistant rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )


def assistant_from_environment(
    prediction_service: EloReferenceService,
    analytics_service: AnalyticsService | None,
) -> AssistantClient | None:
    """Build the optional server assistant only when configuration is present."""
    configured = any(
        os.getenv(name, "").strip()
        for name in ("OPENAI_API_KEY", "FOOTCAST_LLM_MODEL")
    )
    if not configured:
        return None
    if analytics_service is None:
        raise RuntimeError("Assistant requires the analytics service")
    settings = AssistantSettings.from_environment()
    return AssistantClient(
        OpenAIResponsesProvider.from_environment(),
        AssistantTools(prediction_service, analytics_service),
        model=settings.model,
        max_tool_calls=settings.max_tool_calls,
        timeout_seconds=settings.timeout_seconds,
        max_retries=settings.max_retries,
        input_usd_per_million_tokens=settings.input_usd_per_million_tokens,
        output_usd_per_million_tokens=settings.output_usd_per_million_tokens,
    )


def create_chat_router() -> APIRouter:
    router = APIRouter(prefix="/assistant", tags=["assistant"])

    def service(request: Request) -> ChatService:
        return request.app.state.chat_service

    @router.get("/status", response_model=ChatStatusResponse)
    def status(request: Request) -> dict[str, Any]:
        active = service(request)
        return {
            "available": active.available,
            "policy_version": ASSISTANT_POLICY_VERSION,
            "history_turn_limit": MAX_HISTORY_ITEMS // 2,
            "message_character_limit": 1_000,
            "suggested_questions": list(SUGGESTED_QUESTIONS),
        }

    @router.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest, request: Request) -> dict[str, Any]:
        client_ip = request.client.host if request.client else "unknown"
        try:
            session_id, result = service(request).chat(
                payload.message,
                session_id=payload.session_id,
                client_ip=client_ip,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error.args[0])) from error
        except ProviderUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except AssistantClientError as error:
            LOGGER.warning(
                "assistant_request_failed error_type=%s",
                error.__class__.__name__,
            )
            raise HTTPException(
                status_code=502,
                detail="Assistant could not complete the request",
            ) from error
        return {
            "session_id": session_id,
            "answer": result.answer,
            "model": result.model,
            "evidence": [_evidence_dict(item) for item in result.evidence],
            "tool_calls": result.tool_calls,
            "latency_ms": result.latency_ms,
        }

    @router.delete("/sessions/{session_id}", response_model=ResetResponse)
    def reset(session_id: UUID, request: Request, response: Response) -> dict[str, Any]:
        reset_session = service(request).reset(session_id)
        if not reset_session:
            response.status_code = 404
        return {"session_id": session_id, "reset": reset_session}

    return router


def _evidence_dict(evidence: AssistantEvidence) -> dict[str, Any]:
    return {
        "tool_name": evidence.tool_name,
        "answer_mode": evidence.answer_mode,
        "generated_at": evidence.generated_at,
        "source": evidence.source,
        "data_cutoff": evidence.data_cutoff,
        "model_version": evidence.model_version,
        "test_season": evidence.test_season,
        "window": evidence.window,
        "sample_size": evidence.sample_size,
        "documentation_version": evidence.documentation_version,
    }
