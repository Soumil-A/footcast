"""Contract tests for the Phase 9 assistant evaluation benchmark."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

BENCHMARK_PATH = (
    Path(__file__).parents[1] / "evals" / "assistant_questions.jsonl"
)
REQUIRED_FIELDS = {
    "id",
    "category",
    "question",
    "context",
    "expected_tool",
    "expected_arguments",
    "answer_mode",
    "required_evidence",
    "required_facts",
    "risk_tags",
    "acceptable_behavior",
}
TOOL_ARGUMENTS = {
    "get_match_prediction": {"home_team", "away_team", "match_date"},
    "get_team_form": {"team", "limit"},
    "compare_teams": {"team_a", "team_b", "limit"},
    "get_model_explanation": {"topic"},
    "get_metric_definition": {"term"},
}
TOOL_MODES = {
    "get_match_prediction": "prediction",
    "get_team_form": "observed",
    "compare_teams": "observed",
    "get_model_explanation": "explanation",
    "get_metric_definition": "explanation",
}
REQUIRED_TOOL_EVIDENCE = {
    "get_match_prediction": {"model_version", "data_cutoff", "generated_at"},
    "get_team_form": {"window", "data_cutoff", "generated_at"},
    "compare_teams": {"window", "sample_size", "generated_at"},
    "get_model_explanation": {
        "model_version",
        "evidence_source",
        "generated_at",
    },
    "get_metric_definition": {
        "documentation_version",
        "source",
        "generated_at",
    },
}


def _cases() -> list[dict]:
    lines = BENCHMARK_PATH.read_text(encoding="utf-8").splitlines()
    assert all(line.strip() for line in lines)
    return [json.loads(line) for line in lines]


def test_benchmark_has_expected_size_unique_ids_and_balanced_categories() -> None:
    cases = _cases()

    assert 30 <= len(cases) <= 50
    assert len(cases) == 42
    assert len({case["id"] for case in cases}) == len(cases)
    assert Counter(case["category"] for case in cases) == {
        "prediction": 7,
        "team_form": 7,
        "team_comparison": 7,
        "model_explanation": 7,
        "metric_definition": 7,
        "unsupported": 7,
    }


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["id"])
def test_benchmark_case_schema_and_tool_contract(case: dict) -> None:
    assert set(case) == REQUIRED_FIELDS
    assert case["id"]
    assert isinstance(case["question"], str) and case["question"].strip()
    assert isinstance(case["context"], dict)
    assert isinstance(case["expected_arguments"], dict)
    assert isinstance(case["required_evidence"], list)
    assert isinstance(case["required_facts"], list) and case["required_facts"]
    assert isinstance(case["risk_tags"], list)
    assert isinstance(case["acceptable_behavior"], str)
    assert case["acceptable_behavior"].strip()

    tool = case["expected_tool"]
    if tool is None:
        assert case["category"] == "unsupported"
        assert case["answer_mode"] == "refusal"
        assert case["expected_arguments"] == {}
        assert case["required_evidence"] == []
        assert case["risk_tags"]
        return

    assert tool in TOOL_ARGUMENTS
    assert set(case["expected_arguments"]) == TOOL_ARGUMENTS[tool]
    assert case["answer_mode"] == TOOL_MODES[tool]
    assert REQUIRED_TOOL_EVIDENCE[tool].issubset(case["required_evidence"])


def test_benchmark_covers_context_safety_and_model_weaknesses() -> None:
    cases = _cases()
    tags = Counter(
        tag for case in cases for tag in case["risk_tags"]
    )

    assert sum(bool(case["context"]) for case in cases) >= 3
    assert tags["context_follow_up"] >= 3
    assert tags["prompt_injection"] >= 2
    assert tags["betting_safety"] >= 3
    assert tags["draw_limit"] >= 3
    assert tags["unsupported_source"] >= 3


def test_prediction_dates_are_after_the_approved_data_cutoff() -> None:
    for case in _cases():
        if case["expected_tool"] == "get_match_prediction":
            assert case["expected_arguments"]["match_date"] > "2025-05-25"
