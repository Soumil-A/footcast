"""FastAPI application for deterministic FootCast predictions."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from time import perf_counter

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse

from footcast.analytics.portfolio import final_test_evidence
from footcast.analytics.service import AnalyticsInputError, AnalyticsService
from footcast.api.chat import (
    AssistantAnswerer,
    ChatService,
    assistant_from_environment,
    create_chat_router,
)
from footcast.inference.elo_service import (
    REFERENCE_MODEL_VERSION,
    EloReferenceService,
    PredictionInputError,
    load_reference_matches,
)

LOGGER = logging.getLogger("footcast.api")


class PredictionRequest(BaseModel):
    """Only information available before the requested fixture."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    home_team: str = Field(min_length=1, max_length=100)
    away_team: str = Field(min_length=1, max_length=100)
    match_date: date


class PredictionResponse(BaseModel):
    """Versioned three-way probability response."""

    home_team: str
    away_team: str
    match_date: date
    home_win_probability: float = Field(ge=0.0, le=1.0)
    draw_probability: float = Field(ge=0.0, le=1.0)
    away_win_probability: float = Field(ge=0.0, le=1.0)
    predicted_result: str
    home_elo: float
    away_elo: float
    model_version: str
    data_cutoff: date
    intended_use: str
    warning: str


class HealthResponse(BaseModel):
    """Minimal service-readiness response."""

    status: str
    model_version: str
    data_cutoff: date
    holdout_used: bool


class TeamsResponse(BaseModel):
    """Supported team names from approved completed history."""

    teams: list[str]
    count: int
    data_cutoff: date


class FormMatchResponse(BaseModel):
    """One completed fixture from the requested team's perspective."""

    match_date: date
    opponent: str
    venue: str
    goals_for: int
    goals_against: int
    outcome: str
    points: int


class FormSummaryResponse(BaseModel):
    """Aggregate over only the displayed recent matches."""

    matches: int
    wins: int
    draws: int
    losses: int
    points: int
    goals_for: int
    goals_against: int


class TeamFormResponse(BaseModel):
    team: str
    data_cutoff: date
    summary: FormSummaryResponse
    matches: list[FormMatchResponse]


class TeamComparisonResponse(BaseModel):
    home: TeamFormResponse
    away: TeamFormResponse
    home_elo: float
    away_elo: float
    elo_difference: float
    data_cutoff: date


class HeadToHeadMatchResponse(BaseModel):
    match_date: date
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    team_a_outcome: str


class HeadToHeadResponse(BaseModel):
    team_a: str
    team_b: str
    data_cutoff: date
    matches: list[HeadToHeadMatchResponse]


class OutcomeDistributionResponse(BaseModel):
    outcome: str
    matches: int
    share: float = Field(ge=0.0, le=1.0)


class StrengthRankingResponse(BaseModel):
    rank: int
    team: str
    elo: float


class ModelBenchmarkResponse(BaseModel):
    model: str
    accuracy: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    log_loss: float = Field(ge=0.0)


class PortfolioAnalyticsResponse(BaseModel):
    completed_matches: int
    first_match_date: date
    data_cutoff: date
    seasons: list[str]
    season_count: int
    outcome_distribution: list[OutcomeDistributionResponse]
    strength_ranking: list[StrengthRankingResponse]
    test_season: str
    test_matches: int
    benchmarks: list[ModelBenchmarkResponse]
    deployed_elo_recall: dict[str, float]
    deployed_elo_confusion_matrix: list[list[int]]
    class_order: list[str]
    selection_note: str


def create_app(
    service: EloReferenceService | None = None,
    analytics_service: AnalyticsService | None = None,
    assistant_client: AssistantAnswerer | None = None,
    chat_service: ChatService | None = None,
) -> FastAPI:
    """Create an application with injectable deterministic model state."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if service is None:
            approved_matches = load_reference_matches()
            application.state.prediction_service = EloReferenceService(
                approved_matches
            )
            application.state.analytics_service = AnalyticsService(approved_matches)
        else:
            application.state.prediction_service = service
            application.state.analytics_service = analytics_service
        if chat_service is not None:
            application.state.chat_service = chat_service
        else:
            configured_assistant = assistant_client or assistant_from_environment(
                application.state.prediction_service,
                application.state.analytics_service,
            )
            application.state.chat_service = ChatService(configured_assistant)
        yield

    application = FastAPI(
        title="FootCast Prediction API",
        version="0.1.0",
        description=(
            "Educational Premier League probabilities from a transparent "
            "Elo reference model. Not betting advice."
        ),
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def measure_request_time(request: Request, call_next):
        started = perf_counter()
        if request.url.path == "/assistant/chat":
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except ValueError:
                    return JSONResponse(
                        status_code=400,
                        content={"detail": "Invalid Content-Length header"},
                    )
                if declared_size > 8_192:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Assistant request body is too large"},
                    )
            if len(await request.body()) > 8_192:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Assistant request body is too large"},
                )
        response = await call_next(request)
        elapsed_ms = (perf_counter() - started) * 1_000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.3f}"
        LOGGER.info(
            "request_complete method=%s path=%s status=%s elapsed_ms=%.3f",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    application.include_router(create_chat_router())

    def current_service(request: Request) -> EloReferenceService:
        return request.app.state.prediction_service

    def current_analytics(request: Request) -> AnalyticsService:
        active = request.app.state.analytics_service
        if active is None:
            raise HTTPException(status_code=503, detail="Analytics are unavailable")
        return active

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> dict:
        active = current_service(request)
        return {
            "status": "ok",
            "model_version": REFERENCE_MODEL_VERSION,
            "data_cutoff": active.data_cutoff,
            "holdout_used": False,
        }

    @application.get("/teams", response_model=TeamsResponse)
    def teams(request: Request) -> dict:
        active = current_service(request)
        return {
            "teams": list(active.teams),
            "count": len(active.teams),
            "data_cutoff": active.data_cutoff,
        }

    @application.get("/model/info")
    def model_info(request: Request) -> dict:
        return current_service(request).model_info()

    @application.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request) -> dict:
        try:
            prediction = current_service(request).predict(
                payload.home_team,
                payload.away_team,
                payload.match_date,
            )
        except PredictionInputError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return prediction.to_dict()

    @application.get("/analytics/team-form", response_model=TeamFormResponse)
    def team_form(
        request: Request,
        team: str = Query(min_length=1, max_length=100),
        limit: int = Query(default=5, ge=1, le=20),
    ) -> dict:
        try:
            return current_analytics(request).recent_form(team, limit=limit)
        except AnalyticsInputError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.get(
        "/analytics/compare", response_model=TeamComparisonResponse
    )
    def compare_teams(
        request: Request,
        home_team: str = Query(min_length=1, max_length=100),
        away_team: str = Query(min_length=1, max_length=100),
        limit: int = Query(default=5, ge=1, le=20),
    ) -> dict:
        try:
            comparison = current_analytics(request).compare(
                home_team, away_team, limit=limit
            )
            active = current_service(request)
            home_elo = active.rating(home_team)
            away_elo = active.rating(away_team)
        except (AnalyticsInputError, PredictionInputError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {
            **comparison,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_difference": home_elo - away_elo,
        }

    @application.get(
        "/analytics/head-to-head", response_model=HeadToHeadResponse
    )
    def head_to_head(
        request: Request,
        team_a: str = Query(min_length=1, max_length=100),
        team_b: str = Query(min_length=1, max_length=100),
        limit: int = Query(default=10, ge=1, le=20),
    ) -> dict:
        try:
            return current_analytics(request).head_to_head(
                team_a, team_b, limit=limit
            )
        except AnalyticsInputError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.get(
        "/analytics/portfolio", response_model=PortfolioAnalyticsResponse
    )
    def portfolio_analytics(request: Request) -> dict:
        """Expose aggregated history and frozen evaluation evidence."""
        analytics = current_analytics(request).portfolio_overview()
        predictor = current_service(request)
        ranked = sorted(
            (
                {"team": team, "elo": predictor.rating(team)}
                for team in predictor.teams
            ),
            key=lambda item: (-item["elo"], item["team"]),
        )[:10]
        return {
            **analytics,
            "strength_ranking": [
                {"rank": index, **item}
                for index, item in enumerate(ranked, start=1)
            ],
            **final_test_evidence(),
        }

    return application


app = create_app()
