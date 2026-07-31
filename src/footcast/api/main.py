"""FastAPI application for deterministic FootCast predictions."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from footcast.inference.elo_service import (
    REFERENCE_MODEL_VERSION,
    EloReferenceService,
    PredictionInputError,
    load_reference_service,
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


def create_app(service: EloReferenceService | None = None) -> FastAPI:
    """Create an application with injectable deterministic model state."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.prediction_service = service or load_reference_service()
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

    def current_service(request: Request) -> EloReferenceService:
        return request.app.state.prediction_service

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

    return application


app = create_app()
