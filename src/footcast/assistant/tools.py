"""Typed, read-only tools over deterministic FootCast services."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from footcast.analytics.portfolio import final_test_evidence
from footcast.analytics.service import AnalyticsInputError, AnalyticsService
from footcast.assistant.schemas import (
    AssistantToolDescriptor,
    AssistantToolResult,
    CompareTeamsInput,
    CompareTeamsResult,
    ComparisonSampleSize,
    ComparisonTeams,
    DateRange,
    EvidenceFact,
    MatchPredictionInput,
    MatchPredictionResult,
    MetricDefinitionInput,
    MetricDefinitionResult,
    ModelExplanationInput,
    ModelExplanationResult,
    OutcomeProbabilities,
    TeamComparisonMetrics,
    TeamFormInput,
    TeamFormResult,
)
from footcast.inference.elo_service import (
    EloReferenceService,
    PredictionInputError,
)

TOOL_SOURCE = "FootCast approved deterministic services"
DOCUMENTATION_VERSION = "phase-9-checkpoint-2-v1"
FORM_AGGREGATION = (
    "Latest completed matches, newest first; points use 3 for a win, 1 for a "
    "draw, and 0 for a loss."
)

METRIC_DEFINITIONS: dict[str, tuple[str, str, str]] = {
    "elo": (
        "Elo rating",
        "A relative strength rating updated only after each completed result.",
        "A higher rating means stronger historical results in this system; it "
        "does not include injuries, lineups, transfers, or tactics.",
    ),
    "macro_f1": (
        "Macro F1",
        "The unweighted average of the F1 score calculated separately for each "
        "outcome class.",
        "Higher is better. Equal class weighting makes poor draw performance "
        "visible even when draws are less common.",
    ),
    "log_loss": (
        "Multiclass log loss",
        "A probability-quality metric that penalizes confident probability "
        "assigned to the wrong result.",
        "Lower is better. It evaluates the complete probability distribution, "
        "not only the most likely label.",
    ),
    "calibration": (
        "Probability calibration",
        "Agreement between predicted probabilities and observed frequencies.",
        "If many events receive a 70% estimate, about 70% should occur. A model "
        "can be calibrated without having the best classification accuracy.",
    ),
    "brier_score": (
        "Multiclass Brier score",
        "The mean squared difference between predicted class probabilities and "
        "the one-hot encoded result.",
        "Lower is better. The score rewards probability distributions close to "
        "the observed outcome.",
    ),
    "confusion_matrix": (
        "Confusion matrix",
        "A table counting actual result classes by the labels selected by the "
        "model.",
        "FootCast uses actual outcomes as rows and predicted outcomes as columns. "
        "Diagonal cells are correct selections; off-diagonal cells are errors.",
    ),
    "draw_recall": (
        "Draw recall",
        "The fraction of actual draws that the model selected as draws.",
        "It ranges from 0 to 1 and differs from assigning a nonzero draw "
        "probability. The deployed Elo reference had zero draw recall on the "
        "2024-25 test.",
    ),
}


class AssistantToolError(ValueError):
    """Safe tool-routing or deterministic-service failure."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


class AssistantTools:
    """Provider-neutral facade that cannot mutate model or match state."""

    def __init__(
        self,
        prediction_service: EloReferenceService,
        analytics_service: AnalyticsService,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if prediction_service.data_cutoff != analytics_service.data_cutoff:
            raise ValueError("Assistant services must share one data cutoff")
        self._prediction = prediction_service
        self._analytics = analytics_service
        self._clock = clock

    @staticmethod
    def catalog() -> list[AssistantToolDescriptor]:
        """Return model-agnostic tool descriptions and strict input schemas."""
        definitions = (
            (
                "get_match_prediction",
                "Return one future fixture's frozen home/draw/away probabilities.",
                MatchPredictionInput,
            ),
            (
                "get_team_form",
                "Return a team's latest approved completed matches and summary.",
                TeamFormInput,
            ),
            (
                "compare_teams",
                "Compare two teams over the same approved recent-form window.",
                CompareTeamsInput,
            ),
            (
                "get_model_explanation",
                "Return approved model evidence, decisions, and limitations.",
                ModelExplanationInput,
            ),
            (
                "get_metric_definition",
                "Return a versioned definition of one supported evaluation term.",
                MetricDefinitionInput,
            ),
        )
        return [
            AssistantToolDescriptor(
                name=name,
                description=description,
                read_only=True,
                input_schema=input_model.model_json_schema(),
            )
            for name, description, input_model in definitions
        ]

    def execute(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> AssistantToolResult:
        """Validate and execute exactly one named tool."""
        routes: dict[str, tuple[type, Callable]] = {
            "get_match_prediction": (
                MatchPredictionInput,
                self.get_match_prediction,
            ),
            "get_team_form": (TeamFormInput, self.get_team_form),
            "compare_teams": (CompareTeamsInput, self.compare_teams),
            "get_model_explanation": (
                ModelExplanationInput,
                self.get_model_explanation,
            ),
            "get_metric_definition": (
                MetricDefinitionInput,
                self.get_metric_definition,
            ),
        }
        route = routes.get(tool_name)
        if route is None:
            raise AssistantToolError(f"Unknown assistant tool: {tool_name!r}")
        input_model, operation = route
        try:
            payload = input_model.model_validate(arguments)
        except ValidationError as error:
            raise AssistantToolError(
                f"Invalid arguments for {tool_name}: {error.errors(include_url=False)}"
            ) from error
        return operation(payload)

    def get_match_prediction(
        self, payload: MatchPredictionInput
    ) -> MatchPredictionResult:
        try:
            prediction = self._prediction.predict(
                payload.home_team,
                payload.away_team,
                payload.match_date,
            )
        except PredictionInputError as error:
            raise AssistantToolError(str(error)) from error
        return MatchPredictionResult(
            tool_name="get_match_prediction",
            answer_mode="prediction",
            generated_at=self._now(),
            source=TOOL_SOURCE,
            home_team=prediction.home_team,
            away_team=prediction.away_team,
            match_date=prediction.match_date,
            probabilities=OutcomeProbabilities(
                home_win=prediction.home_win_probability,
                draw=prediction.draw_probability,
                away_win=prediction.away_win_probability,
            ),
            predicted_result=prediction.predicted_result,
            home_elo=prediction.home_elo,
            away_elo=prediction.away_elo,
            model_version=prediction.model_version,
            data_cutoff=prediction.data_cutoff,
            intended_use=prediction.intended_use,
            warning=prediction.warning,
        )

    def get_team_form(self, payload: TeamFormInput) -> TeamFormResult:
        try:
            form = self._analytics.recent_form(payload.team, limit=payload.limit)
        except AnalyticsInputError as error:
            raise AssistantToolError(str(error)) from error
        matches = form["matches"]
        match_dates = [match["match_date"] for match in matches]
        return TeamFormResult(
            tool_name="get_team_form",
            answer_mode="observed",
            generated_at=self._now(),
            source=TOOL_SOURCE,
            team=form["team"],
            window=payload.limit,
            date_range=DateRange(
                start=min(match_dates) if match_dates else None,
                end=max(match_dates) if match_dates else None,
            ),
            aggregation=FORM_AGGREGATION,
            data_cutoff=form["data_cutoff"],
            summary=form["summary"],
            matches=matches,
        )

    def compare_teams(self, payload: CompareTeamsInput) -> CompareTeamsResult:
        try:
            comparison = self._analytics.compare(
                payload.team_a,
                payload.team_b,
                limit=payload.limit,
            )
            team_a_elo = self._prediction.rating(payload.team_a)
            team_b_elo = self._prediction.rating(payload.team_b)
        except (AnalyticsInputError, PredictionInputError) as error:
            raise AssistantToolError(str(error)) from error
        team_a = comparison["home"]
        team_b = comparison["away"]
        return CompareTeamsResult(
            tool_name="compare_teams",
            answer_mode="observed",
            generated_at=self._now(),
            source=TOOL_SOURCE,
            teams=ComparisonTeams(
                team_a=team_a["team"],
                team_b=team_b["team"],
            ),
            window=payload.limit,
            sample_size=ComparisonSampleSize(
                team_a_matches=team_a["summary"]["matches"],
                team_b_matches=team_b["summary"]["matches"],
            ),
            metrics=TeamComparisonMetrics(
                team_a_form=team_a["summary"],
                team_b_form=team_b["summary"],
                team_a_elo=team_a_elo,
                team_b_elo=team_b_elo,
                elo_difference_team_a_minus_team_b=team_a_elo - team_b_elo,
            ),
            data_cutoff=comparison["data_cutoff"],
        )

    def get_model_explanation(
        self, payload: ModelExplanationInput
    ) -> ModelExplanationResult:
        info = self._prediction.model_info()
        evidence = final_test_evidence()
        title, summary, facts, source = self._model_explanation(
            payload.topic, info, evidence
        )
        return ModelExplanationResult(
            tool_name="get_model_explanation",
            answer_mode="explanation",
            generated_at=self._now(),
            source=TOOL_SOURCE,
            evidence_source=source,
            topic=payload.topic,
            title=title,
            summary=summary,
            facts=[EvidenceFact(label=label, value=value) for label, value in facts],
            model_version=info["model_version"],
            test_season=evidence["test_season"],
            data_cutoff=info["data_cutoff"],
            limitations=info["limitations"],
        )

    def get_metric_definition(
        self, payload: MetricDefinitionInput
    ) -> MetricDefinitionResult:
        display_name, definition, interpretation = METRIC_DEFINITIONS[payload.term]
        return MetricDefinitionResult(
            tool_name="get_metric_definition",
            answer_mode="explanation",
            generated_at=self._now(),
            source="docs/model_card.md and FootCast evaluation documentation",
            term=payload.term,
            display_name=display_name,
            definition=definition,
            interpretation=interpretation,
            documentation_version=DOCUMENTATION_VERSION,
        )

    def _now(self) -> datetime:
        timestamp = self._clock()
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("Assistant tool clock must return a timezone-aware value")
        return timestamp

    @staticmethod
    def _model_explanation(
        topic: str,
        info: dict[str, Any],
        evidence: dict[str, Any],
    ) -> tuple[str, str, list[tuple[str, str | int | float | bool]], str]:
        benchmark = {item["model"]: item for item in evidence["benchmarks"]}
        elo = benchmark["Elo (deployed)"]
        forest = benchmark["Frozen Random Forest"]
        explanations = {
            "model_selection": (
                "Why Elo is deployed",
                "The frozen Random Forest did not show a stable final-test "
                "advantage, so FootCast uses the simpler transparent Elo reference.",
                [
                    ("Elo test accuracy", elo["accuracy"]),
                    ("Elo test log loss", elo["log_loss"]),
                    ("Forest test accuracy", forest["accuracy"]),
                    ("Forest test log loss", forest["log_loss"]),
                ],
                "reports/final_test_results.json and docs/model_card.md",
            ),
            "draw_recall": (
                "Draw-recognition limitation",
                "The reference model assigns a historical draw probability but "
                "did not select draws as its highest-probability class on the test.",
                [
                    ("Elo draw recall", evidence["deployed_elo_recall"]["draw"]),
                    ("Test matches", evidence["test_matches"]),
                    ("Draw probability definition", "historical approved-match rate"),
                ],
                "reports/final_test_results.json and models/elo_reference_spec.json",
            ),
            "deployed_features": (
                "Inputs to the deployed reference",
                "The live model uses replayed Elo ratings, a fixed home-advantage "
                "constant, and the approved-history draw rate.",
                [
                    ("Model type", info["model_type"]),
                    ("Home advantage", info["elo_config"]["home_advantage"]),
                    ("Historical draw rate", info["draw_probability"]),
                    ("Completed matches", info["completed_matches"]),
                ],
                "models/elo_reference_spec.json",
            ),
            "limitations": (
                "Approved model limitations",
                "FootCast is a historical educational estimate and omits live "
                "injuries, lineups, transfers, and tactics.",
                [
                    ("Data cutoff", info["data_cutoff"].isoformat()),
                    ("Intended use", info["intended_use"]),
                    ("Holdout seasons used", bool(info["holdout_seasons_used"])),
                ],
                "docs/model_card.md and models/elo_reference_spec.json",
            ),
            "final_test": (
                "Untouched 2024-25 Elo test",
                "FootCast reports the frozen test result without using it to "
                "retune the deployed reference.",
                [
                    ("Test matches", evidence["test_matches"]),
                    ("Accuracy", elo["accuracy"]),
                    ("Macro F1", elo["macro_f1"]),
                    ("Log loss", elo["log_loss"]),
                    ("Draw recall", evidence["deployed_elo_recall"]["draw"]),
                ],
                "reports/final_test_results.json",
            ),
            "calibration_decision": (
                "Random Forest calibration decision",
                "Forward-only validation found that sigmoid and isotonic "
                "calibration worsened probability quality, so neither was retained.",
                [
                    ("Uncalibrated mean log loss", 0.994),
                    ("Sigmoid mean log loss", 1.003),
                    ("Isotonic mean log loss", 1.166),
                    ("Selected method", "none"),
                ],
                "reports/calibration_results.json and docs/calibration.md",
            ),
            "analytics_vs_prediction": (
                "Descriptive analytics versus prediction",
                "Recent form and head-to-head charts explain historical context "
                "but do not change the deployed Elo probability calculation.",
                [
                    ("Prediction model", info["model_type"]),
                    ("Dashboard form input to model", False),
                    ("LLM adjustment to probabilities", False),
                ],
                "docs/dashboard.md and models/elo_reference_spec.json",
            ),
        }
        return explanations[topic]
