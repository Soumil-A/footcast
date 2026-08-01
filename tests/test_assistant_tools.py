"""Tests for deterministic, read-only Phase 9 assistant tools."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from footcast.analytics.service import AnalyticsService
from footcast.assistant.schemas import (
    CompareTeamsInput,
    CompareTeamsResult,
    MatchPredictionInput,
    MatchPredictionResult,
    MetricDefinitionInput,
    MetricDefinitionResult,
    ModelExplanationInput,
    ModelExplanationResult,
    TeamFormInput,
    TeamFormResult,
)
from footcast.assistant.tools import AssistantToolError, AssistantTools
from footcast.features.elo import EloConfig
from footcast.inference.elo_service import EloReferenceService

FIXED_NOW = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
BENCHMARK_PATH = (
    Path(__file__).parents[1] / "evals" / "assistant_questions.jsonl"
)
INPUT_MODELS = {
    "get_match_prediction": MatchPredictionInput,
    "get_team_form": TeamFormInput,
    "compare_teams": CompareTeamsInput,
    "get_model_explanation": ModelExplanationInput,
    "get_metric_definition": MetricDefinitionInput,
}
RESULT_MODELS = {
    "get_match_prediction": MatchPredictionResult,
    "get_team_form": TeamFormResult,
    "compare_teams": CompareTeamsResult,
    "get_model_explanation": ModelExplanationResult,
    "get_metric_definition": MetricDefinitionResult,
}


def _matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": "2023-24",
                "split": "validation",
                "match_date": "2024-01-01",
                "home_team": "Alpha",
                "away_team": "Beta",
                "full_time_home_goals": 2,
                "full_time_away_goals": 0,
                "result": "home_win",
            },
            {
                "season": "2023-24",
                "split": "validation",
                "match_date": "2024-01-08",
                "home_team": "Gamma",
                "away_team": "Alpha",
                "full_time_home_goals": 1,
                "full_time_away_goals": 1,
                "result": "draw",
            },
            {
                "season": "2023-24",
                "split": "validation",
                "match_date": "2024-01-15",
                "home_team": "Beta",
                "away_team": "Alpha",
                "full_time_home_goals": 3,
                "full_time_away_goals": 1,
                "result": "home_win",
            },
            {
                "season": "2023-24",
                "split": "validation",
                "match_date": "2024-01-22",
                "home_team": "Alpha",
                "away_team": "Gamma",
                "full_time_home_goals": 2,
                "full_time_away_goals": 1,
                "result": "home_win",
            },
        ]
    )


@pytest.fixture
def services() -> tuple[EloReferenceService, AnalyticsService]:
    matches = _matches()
    return (
        EloReferenceService(
            matches, elo_config=EloConfig(home_advantage=0.0)
        ),
        AnalyticsService(matches),
    )


@pytest.fixture
def tools(
    services: tuple[EloReferenceService, AnalyticsService],
) -> AssistantTools:
    prediction, analytics = services
    return AssistantTools(prediction, analytics, clock=lambda: FIXED_NOW)


def test_catalog_exposes_five_strict_read_only_tools() -> None:
    catalog = AssistantTools.catalog()

    assert [tool.name for tool in catalog] == list(INPUT_MODELS)
    assert all(tool.read_only is True for tool in catalog)
    assert all(tool.input_schema["additionalProperties"] is False for tool in catalog)
    assert all(tool.description for tool in catalog)


def test_prediction_tool_returns_probabilities_provenance_and_no_mutation(
    tools: AssistantTools,
    services: tuple[EloReferenceService, AnalyticsService],
) -> None:
    prediction, _ = services
    ratings_before = {team: prediction.rating(team) for team in prediction.teams}

    result = tools.get_match_prediction(
        MatchPredictionInput(
            home_team="Alpha",
            away_team="Beta",
            match_date=date(2024, 2, 1),
        )
    )

    probabilities = result.probabilities
    assert result.tool_name == "get_match_prediction"
    assert result.answer_mode == "prediction"
    assert result.generated_at == FIXED_NOW
    assert result.data_cutoff == date(2024, 1, 22)
    assert sum(
        [probabilities.home_win, probabilities.draw, probabilities.away_win]
    ) == pytest.approx(1.0)
    assert result.model_version
    assert result.warning
    ratings_after = {
        team: prediction.rating(team) for team in prediction.teams
    }
    assert ratings_after == ratings_before


def test_team_form_tool_returns_window_dates_and_aggregation(
    tools: AssistantTools,
) -> None:
    result = tools.get_team_form(TeamFormInput(team="Alpha", limit=3))

    assert result.team == "Alpha"
    assert result.window == 3
    assert result.summary.matches == 3
    assert result.date_range.start == date(2024, 1, 8)
    assert result.date_range.end == date(2024, 1, 22)
    assert [match.match_date for match in result.matches] == [
        date(2024, 1, 22),
        date(2024, 1, 15),
        date(2024, 1, 8),
    ]
    assert "newest first" in result.aggregation
    assert result.generated_at == FIXED_NOW


def test_comparison_tool_returns_equal_windows_ratings_and_sample_sizes(
    tools: AssistantTools,
) -> None:
    result = tools.compare_teams(
        CompareTeamsInput(team_a="Alpha", team_b="Beta", limit=2)
    )

    assert result.teams.team_a == "Alpha"
    assert result.teams.team_b == "Beta"
    assert result.window == 2
    assert result.sample_size.team_a_matches == 2
    assert result.sample_size.team_b_matches == 2
    assert result.metrics.elo_difference_team_a_minus_team_b == pytest.approx(
        result.metrics.team_a_elo - result.metrics.team_b_elo
    )
    assert result.data_cutoff == date(2024, 1, 22)


@pytest.mark.parametrize(
    "topic",
    [
        "model_selection",
        "draw_recall",
        "deployed_features",
        "limitations",
        "final_test",
        "calibration_decision",
        "analytics_vs_prediction",
    ],
)
def test_model_explanation_topics_are_sourced_and_versioned(
    tools: AssistantTools, topic: str
) -> None:
    result = tools.get_model_explanation(ModelExplanationInput(topic=topic))

    assert result.topic == topic
    assert result.facts
    assert result.evidence_source
    assert result.model_version == "footcast-elo-v2-reference"
    assert result.test_season == "2024-25"
    assert result.limitations
    assert result.generated_at == FIXED_NOW


@pytest.mark.parametrize(
    "term",
    [
        "elo",
        "macro_f1",
        "log_loss",
        "calibration",
        "brier_score",
        "confusion_matrix",
        "draw_recall",
    ],
)
def test_metric_definitions_are_versioned(tools: AssistantTools, term: str) -> None:
    result = tools.get_metric_definition(MetricDefinitionInput(term=term))

    assert result.term == term
    assert result.definition
    assert result.interpretation
    assert result.documentation_version
    assert result.source
    assert result.generated_at == FIXED_NOW


def test_calibration_explanation_matches_tracked_report(
    tools: AssistantTools,
) -> None:
    report_path = (
        Path(__file__).parents[1] / "reports" / "calibration_results.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidates = {
        item["method"]: item
        for item in report["calibration_selection"]["candidates"]
    }
    result = tools.get_model_explanation(
        ModelExplanationInput(topic="calibration_decision")
    )
    facts = {fact.label: fact.value for fact in result.facts}

    assert facts["Uncalibrated mean log loss"] == pytest.approx(
        candidates["uncalibrated"]["mean_log_loss"], abs=0.0005
    )
    assert facts["Sigmoid mean log loss"] == pytest.approx(
        candidates["sigmoid"]["mean_log_loss"], abs=0.0005
    )
    assert facts["Isotonic mean log loss"] == pytest.approx(
        candidates["isotonic"]["mean_log_loss"], abs=0.0005
    )


@pytest.mark.parametrize(
    ("tool_name", "arguments", "message"),
    [
        ("unknown_tool", {}, "Unknown assistant tool"),
        ("get_team_form", {"team": "Alpha", "limit": 21}, "Invalid arguments"),
        (
            "get_team_form",
            {"team": "Alpha", "limit": 5, "result": "win"},
            "Invalid arguments",
        ),
        ("get_team_form", {"team": "Unknown", "limit": 5}, "Unknown team"),
        (
            "get_match_prediction",
            {
                "home_team": "Alpha",
                "away_team": "Beta",
                "match_date": "2024-01-22",
            },
            "must be after data cutoff",
        ),
        (
            "compare_teams",
            {"team_a": "Alpha", "team_b": "Alpha", "limit": 5},
            "must be different",
        ),
    ],
)
def test_dispatcher_rejects_unknown_invalid_or_unsupported_inputs(
    tools: AssistantTools,
    tool_name: str,
    arguments: dict,
    message: str,
) -> None:
    with pytest.raises(AssistantToolError, match=message):
        tools.execute(tool_name, arguments)


def test_dispatcher_returns_typed_json_serializable_result(
    tools: AssistantTools,
) -> None:
    result = tools.execute(
        "get_metric_definition", {"term": "calibration"}
    )

    assert isinstance(result, MetricDefinitionResult)
    payload = result.model_dump(mode="json")
    assert payload["generated_at"] == "2026-08-01T12:00:00Z"
    json.dumps(payload)


def test_services_must_share_one_data_cutoff(
    services: tuple[EloReferenceService, AnalyticsService],
) -> None:
    prediction, _ = services
    shorter = AnalyticsService(_matches().iloc[:-1])

    with pytest.raises(ValueError, match="share one data cutoff"):
        AssistantTools(prediction, shorter)


def test_generated_timestamp_must_be_timezone_aware(
    services: tuple[EloReferenceService, AnalyticsService],
) -> None:
    prediction, analytics = services
    tools = AssistantTools(
        prediction,
        analytics,
        clock=lambda: datetime(2026, 8, 1, 12, 0),
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        tools.get_metric_definition(MetricDefinitionInput(term="elo"))


def test_benchmark_arguments_match_tool_schemas_and_evidence_fields() -> None:
    cases = [
        json.loads(line)
        for line in BENCHMARK_PATH.read_text(encoding="utf-8").splitlines()
    ]

    for case in cases:
        tool_name = case["expected_tool"]
        if tool_name is None:
            continue
        INPUT_MODELS[tool_name].model_validate(case["expected_arguments"])
        output_fields = set(RESULT_MODELS[tool_name].model_fields)
        assert set(case["required_evidence"]).issubset(output_fields)
