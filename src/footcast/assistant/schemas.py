"""Strict schemas for provider-neutral, read-only assistant tools."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ModelExplanationTopic = Literal[
    "model_selection",
    "draw_recall",
    "deployed_features",
    "limitations",
    "final_test",
    "calibration_decision",
    "analytics_vs_prediction",
]
MetricTerm = Literal[
    "elo",
    "macro_f1",
    "log_loss",
    "calibration",
    "brier_score",
    "confusion_matrix",
    "draw_recall",
]


class StrictModel(BaseModel):
    """Reject unknown fields so tool calls cannot silently expand scope."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class MatchPredictionInput(StrictModel):
    home_team: str = Field(min_length=1, max_length=100)
    away_team: str = Field(min_length=1, max_length=100)
    match_date: date


class TeamFormInput(StrictModel):
    team: str = Field(min_length=1, max_length=100)
    limit: int = Field(default=5, ge=1, le=20)


class CompareTeamsInput(StrictModel):
    team_a: str = Field(min_length=1, max_length=100)
    team_b: str = Field(min_length=1, max_length=100)
    limit: int = Field(default=5, ge=1, le=20)


class ModelExplanationInput(StrictModel):
    topic: ModelExplanationTopic


class MetricDefinitionInput(StrictModel):
    term: MetricTerm


class OutcomeProbabilities(StrictModel):
    home_win: float = Field(ge=0.0, le=1.0)
    draw: float = Field(ge=0.0, le=1.0)
    away_win: float = Field(ge=0.0, le=1.0)


class MatchPredictionResult(StrictModel):
    tool_name: Literal["get_match_prediction"]
    answer_mode: Literal["prediction"]
    generated_at: datetime
    source: str
    home_team: str
    away_team: str
    match_date: date
    probabilities: OutcomeProbabilities
    predicted_result: str
    home_elo: float
    away_elo: float
    model_version: str
    data_cutoff: date
    intended_use: str
    warning: str


class FormMatch(StrictModel):
    match_date: date
    opponent: str
    venue: Literal["home", "away"]
    goals_for: int = Field(ge=0)
    goals_against: int = Field(ge=0)
    outcome: Literal["win", "draw", "loss"]
    points: int = Field(ge=0, le=3)


class FormSummary(StrictModel):
    matches: int = Field(ge=0, le=20)
    wins: int = Field(ge=0, le=20)
    draws: int = Field(ge=0, le=20)
    losses: int = Field(ge=0, le=20)
    points: int = Field(ge=0, le=60)
    goals_for: int = Field(ge=0)
    goals_against: int = Field(ge=0)


class DateRange(StrictModel):
    start: date | None
    end: date | None


class TeamFormResult(StrictModel):
    tool_name: Literal["get_team_form"]
    answer_mode: Literal["observed"]
    generated_at: datetime
    source: str
    team: str
    window: int = Field(ge=1, le=20)
    date_range: DateRange
    aggregation: str
    data_cutoff: date
    summary: FormSummary
    matches: list[FormMatch]


class ComparisonSampleSize(StrictModel):
    team_a_matches: int = Field(ge=0, le=20)
    team_b_matches: int = Field(ge=0, le=20)


class ComparisonTeams(StrictModel):
    team_a: str
    team_b: str


class TeamComparisonMetrics(StrictModel):
    team_a_form: FormSummary
    team_b_form: FormSummary
    team_a_elo: float
    team_b_elo: float
    elo_difference_team_a_minus_team_b: float


class CompareTeamsResult(StrictModel):
    tool_name: Literal["compare_teams"]
    answer_mode: Literal["observed"]
    generated_at: datetime
    source: str
    teams: ComparisonTeams
    window: int = Field(ge=1, le=20)
    sample_size: ComparisonSampleSize
    metrics: TeamComparisonMetrics
    data_cutoff: date


class EvidenceFact(StrictModel):
    label: str
    value: str | int | float | bool


class ModelExplanationResult(StrictModel):
    tool_name: Literal["get_model_explanation"]
    answer_mode: Literal["explanation"]
    generated_at: datetime
    source: str
    evidence_source: str
    topic: ModelExplanationTopic
    title: str
    summary: str
    facts: list[EvidenceFact]
    model_version: str
    test_season: str
    data_cutoff: date
    limitations: list[str]


class MetricDefinitionResult(StrictModel):
    tool_name: Literal["get_metric_definition"]
    answer_mode: Literal["explanation"]
    generated_at: datetime
    source: str
    term: MetricTerm
    display_name: str
    definition: str
    interpretation: str
    documentation_version: str


class AssistantToolDescriptor(StrictModel):
    name: str
    description: str
    read_only: Literal[True]
    input_schema: dict[str, Any]


AssistantToolInput = (
    MatchPredictionInput
    | TeamFormInput
    | CompareTeamsInput
    | ModelExplanationInput
    | MetricDefinitionInput
)
AssistantToolResult = (
    MatchPredictionResult
    | TeamFormResult
    | CompareTeamsResult
    | ModelExplanationResult
    | MetricDefinitionResult
)
