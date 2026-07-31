"""Tests for completed-history analytics and API contracts."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from footcast.analytics.portfolio import final_test_evidence
from footcast.analytics.service import AnalyticsInputError, AnalyticsService
from footcast.api.main import create_app
from footcast.features.elo import EloConfig
from footcast.inference.elo_service import EloReferenceService


def _matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": "2023-24",
                "split": "train",
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
                "split": "test",
                "match_date": "2024-01-15",
                "home_team": "Beta",
                "away_team": "Alpha",
                "full_time_home_goals": 3,
                "full_time_away_goals": 1,
                "result": "home_win",
            },
        ]
    )


@pytest.fixture
def analytics() -> AnalyticsService:
    return AnalyticsService(_matches())


def test_recent_form_uses_team_perspective_and_latest_first(
    analytics: AnalyticsService,
) -> None:
    form = analytics.recent_form("Alpha", limit=2)

    assert form["data_cutoff"] == date(2024, 1, 15)
    assert form["summary"] == {
        "matches": 2,
        "wins": 0,
        "draws": 1,
        "losses": 1,
        "points": 1,
        "goals_for": 2,
        "goals_against": 4,
    }
    assert [match["match_date"] for match in form["matches"]] == [
        date(2024, 1, 15),
        date(2024, 1, 8),
    ]
    assert form["matches"][0] == {
        "match_date": date(2024, 1, 15),
        "opponent": "Beta",
        "venue": "away",
        "goals_for": 1,
        "goals_against": 3,
        "outcome": "loss",
        "points": 0,
    }


def test_compare_and_head_to_head_preserve_orientation(
    analytics: AnalyticsService,
) -> None:
    comparison = analytics.compare("Alpha", "Beta", limit=1)
    meetings = analytics.head_to_head("Alpha", "Beta", limit=10)

    assert comparison["home"]["matches"][0]["opponent"] == "Beta"
    assert comparison["away"]["matches"][0]["opponent"] == "Alpha"
    assert meetings["matches"][0] == {
        "match_date": date(2024, 1, 15),
        "home_team": "Beta",
        "away_team": "Alpha",
        "home_goals": 3,
        "away_goals": 1,
        "team_a_outcome": "loss",
    }
    assert meetings["matches"][1]["team_a_outcome"] == "win"


def test_portfolio_overview_summarizes_approved_history(
    analytics: AnalyticsService,
) -> None:
    overview = analytics.portfolio_overview()

    assert overview["completed_matches"] == 3
    assert overview["first_match_date"] == date(2024, 1, 1)
    assert overview["data_cutoff"] == date(2024, 1, 15)
    assert overview["seasons"] == ["2023-24"]
    assert overview["season_count"] == 1
    assert sum(item["matches"] for item in overview["outcome_distribution"]) == 3
    total_share = sum(
        item["share"] for item in overview["outcome_distribution"]
    )
    assert total_share == pytest.approx(1.0)


def test_portfolio_model_evidence_matches_tracked_final_test() -> None:
    report_path = Path(__file__).parents[1] / "reports" / "final_test_results.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evidence = final_test_evidence()

    source_names = {
        "Majority class": "Majority class",
        "Elo (deployed)": "Elo",
        "Logistic regression": "Logistic regression",
        "Frozen Random Forest": "Frozen Random Forest",
    }
    for benchmark in evidence["benchmarks"]:
        source = report["results"][source_names[benchmark["model"]]]
        assert benchmark["accuracy"] == source["accuracy"]
        assert benchmark["macro_f1"] == source["macro_f1"]
        assert benchmark["log_loss"] == source["log_loss"]
    assert evidence["deployed_elo_confusion_matrix"] == report["results"]["Elo"][
        "confusion_matrix"
    ]


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda service: service.recent_form("Unknown"), "Unknown team"),
        (lambda service: service.compare("Alpha", "Alpha"), "must be different"),
        (lambda service: service.recent_form("Alpha", limit=0), "between 1 and 20"),
    ],
)
def test_analytics_reject_invalid_requests(
    analytics: AnalyticsService, operation, message: str
) -> None:
    with pytest.raises(AnalyticsInputError, match=message):
        operation(analytics)


def test_analytics_rejects_holdout() -> None:
    matches = _matches()
    matches.loc[0, "split"] = "holdout"

    with pytest.raises(ValueError, match="Holdout"):
        AnalyticsService(matches)


def test_analytics_endpoints_return_typed_comparison() -> None:
    matches = _matches()
    analytics_service = AnalyticsService(matches)
    predictor = EloReferenceService(
        matches, elo_config=EloConfig(home_advantage=0.0)
    )

    with TestClient(create_app(predictor, analytics_service)) as client:
        form = client.get(
            "/analytics/team-form", params={"team": "Alpha", "limit": 2}
        )
        comparison = client.get(
            "/analytics/compare",
            params={"home_team": "Alpha", "away_team": "Beta"},
        )
        meetings = client.get(
            "/analytics/head-to-head",
            params={"team_a": "Alpha", "team_b": "Beta"},
        )
        portfolio = client.get("/analytics/portfolio")
        invalid = client.get(
            "/analytics/team-form", params={"team": "Unknown", "limit": 5}
        )
        invalid_limit = client.get(
            "/analytics/team-form", params={"team": "Alpha", "limit": 21}
        )

    assert form.status_code == 200
    assert form.json()["summary"]["points"] == 1
    assert comparison.status_code == 200
    assert comparison.json()["home_elo"] == pytest.approx(
        predictor.rating("Alpha")
    )
    assert comparison.json()["elo_difference"] == pytest.approx(
        predictor.rating("Alpha") - predictor.rating("Beta")
    )
    assert meetings.status_code == 200
    assert len(meetings.json()["matches"]) == 2
    assert portfolio.status_code == 200
    assert portfolio.json()["completed_matches"] == 3
    assert portfolio.json()["strength_ranking"][0]["rank"] == 1
    assert portfolio.json()["test_season"] == "2024-25"
    assert portfolio.json()["benchmarks"][1]["model"] == "Elo (deployed)"
    assert invalid.status_code == 422
    assert invalid.json()["detail"] == "Unknown team: 'Unknown'"
    assert invalid_limit.status_code == 422
