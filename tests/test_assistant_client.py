"""Offline tests for the bounded Phase 9 assistant client."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from pydantic import BaseModel

from footcast.assistant.client import (
    AssistantClient,
    ProviderResponseError,
    ProviderTurn,
    ProviderUnavailableError,
    ProviderUsage,
    ToolCallLimitError,
    ToolCallProtocolError,
    TransientProviderError,
    responses_tool_catalog,
)
from footcast.assistant.policy import ASSISTANT_INSTRUCTIONS
from footcast.assistant.schemas import AssistantToolDescriptor


class FakeResult(BaseModel):
    tool_name: str
    value: str = "approved evidence"


class FakeTools:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @staticmethod
    def catalog() -> list[AssistantToolDescriptor]:
        return [
            AssistantToolDescriptor(
                name="get_metric_definition",
                description="Return one approved metric definition.",
                read_only=True,
                input_schema={
                    "type": "object",
                    "properties": {
                        "term": {"type": "string", "enum": ["log_loss"]},
                        "detail": {"type": "integer", "default": 1},
                    },
                    "required": ["term"],
                    "additionalProperties": False,
                },
            )
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> FakeResult:
        self.calls.append((name, arguments))
        if set(arguments) != {"term", "detail"}:
            raise ValueError("invalid arguments")
        return FakeResult(tool_name=name)


class ScriptedProvider:
    def __init__(self, script: Sequence[ProviderTurn | Exception]) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []

    def create_response(
        self,
        *,
        model: str,
        instructions: str,
        input_items: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        timeout_seconds: float,
    ) -> ProviderTurn:
        self.requests.append(
            {
                "model": model,
                "instructions": instructions,
                "input_items": [dict(item) for item in input_items],
                "tools": [dict(tool) for tool in tools],
                "timeout_seconds": timeout_seconds,
            }
        )
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def tool_turn(call_id: str = "call_1") -> ProviderTurn:
    return ProviderTurn(
        response_id="resp_tools",
        output_items=(
            {"id": "reason_1", "type": "reasoning", "content": []},
            {
                "type": "function_call",
                "call_id": call_id,
                "name": "get_metric_definition",
                "arguments": json.dumps({"term": "log_loss", "detail": 1}),
            },
        ),
        usage=ProviderUsage(input_tokens=10, output_tokens=3, total_tokens=13),
    )


def final_turn(
    answer: str = "Log loss is lower when probabilities improve.",
) -> ProviderTurn:
    return ProviderTurn(
        response_id="resp_final",
        output_items=(
            {"type": "message", "role": "assistant", "content": []},
        ),
        output_text=answer,
        usage=ProviderUsage(input_tokens=20, output_tokens=5, total_tokens=25),
    )


def test_direct_answer_requires_no_tool() -> None:
    provider = ScriptedProvider([final_turn("Please choose a team.")])
    tools = FakeTools()

    result = AssistantClient(provider, tools, model="test-model").answer("Help")

    assert result.answer == "Please choose a team."
    assert result.tool_calls == 0
    assert tools.calls == []


def test_tool_loop_preserves_output_items_and_returns_matching_call_output() -> None:
    provider = ScriptedProvider([tool_turn(), final_turn()])
    tools = FakeTools()

    result = AssistantClient(provider, tools, model="test-model").answer(
        "What is log loss?"
    )

    assert result.tool_names == ("get_metric_definition",)
    assert tools.calls == [("get_metric_definition", {"term": "log_loss", "detail": 1})]
    second_input = provider.requests[1]["input_items"]
    assert second_input[1]["type"] == "reasoning"
    assert second_input[2]["type"] == "function_call"
    assert second_input[3]["type"] == "function_call_output"
    assert second_input[3]["call_id"] == "call_1"
    assert "approved evidence" in second_input[3]["output"]


def test_responses_catalog_uses_strict_required_schema() -> None:
    catalog = responses_tool_catalog(FakeTools.catalog())

    assert catalog[0]["strict"] is True
    assert catalog[0]["parameters"]["additionalProperties"] is False
    assert catalog[0]["parameters"]["required"] == ["term", "detail"]
    assert "default" not in catalog[0]["parameters"]["properties"]["detail"]


def test_request_is_configured_and_does_not_contain_a_key() -> None:
    provider = ScriptedProvider([final_turn()])
    AssistantClient(
        provider,
        FakeTools(),
        model="chosen-model",
        timeout_seconds=7.5,
    ).answer("Define log loss")

    request = provider.requests[0]
    assert request["model"] == "chosen-model"
    assert request["timeout_seconds"] == 7.5
    assert request["instructions"] == ASSISTANT_INSTRUCTIONS
    assert "api_key" not in request
    assert set(request) == {
        "model",
        "instructions",
        "input_items",
        "tools",
        "timeout_seconds",
    }


def test_usage_cost_and_latency_are_aggregated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ticks = iter([1.0, 1.125])
    provider = ScriptedProvider([tool_turn(), final_turn()])
    caplog.set_level(logging.INFO, logger="footcast.assistant.client")

    result = AssistantClient(
        provider,
        FakeTools(),
        model="test-model",
        input_usd_per_million_tokens=2.0,
        output_usd_per_million_tokens=8.0,
        clock=lambda: next(ticks),
    ).answer("Define log loss")

    assert result.input_tokens == 30
    assert result.output_tokens == 8
    assert result.total_tokens == 38
    assert result.estimated_cost_usd == 0.000124
    assert result.latency_ms == 125
    assert caplog.records[-1].assistant_tool_calls == 1
    assert not hasattr(caplog.records[-1], "assistant_question")


def test_transient_failures_retry_with_exponential_backoff() -> None:
    provider = ScriptedProvider(
        [
            TransientProviderError("timeout"),
            TransientProviderError("rate"),
            final_turn(),
        ]
    )
    delays: list[float] = []

    result = AssistantClient(
        provider,
        FakeTools(),
        model="test-model",
        max_retries=2,
        retry_backoff_seconds=0.5,
        sleep=delays.append,
    ).answer("Help")

    assert result.provider_responses == 1
    assert delays == [0.5, 1.0]
    assert len(provider.requests) == 3


def test_transient_failure_exhaustion_returns_safe_error() -> None:
    provider = ScriptedProvider(
        [TransientProviderError("secret upstream detail")] * 2
    )
    with pytest.raises(
        ProviderUnavailableError, match="temporarily unavailable"
    ) as captured:
        AssistantClient(
            provider,
            FakeTools(),
            model="test-model",
            max_retries=1,
            sleep=lambda _: None,
        ).answer("Help")
    assert "secret" not in str(captured.value)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ("not-json", "malformed arguments"),
        ("[]", "non-object arguments"),
        (json.dumps({"term": "log_loss"}), "invalid arguments"),
    ],
)
def test_invalid_tool_arguments_fail_closed(arguments: str, message: str) -> None:
    turn = tool_turn()
    call = dict(turn.output_items[1])
    call["arguments"] = arguments
    provider = ScriptedProvider(
        [ProviderTurn(response_id="bad", output_items=(call,))]
    )
    with pytest.raises(ToolCallProtocolError, match=message):
        AssistantClient(provider, FakeTools(), model="test-model").answer("Help")


def test_unknown_tool_fails_closed() -> None:
    call = dict(tool_turn().output_items[1])
    call["name"] = "read_environment"
    provider = ScriptedProvider(
        [ProviderTurn(response_id="bad", output_items=(call,))]
    )
    with pytest.raises(ToolCallProtocolError, match="unknown tool"):
        AssistantClient(provider, FakeTools(), model="test-model").answer("Help")


def test_maximum_tool_call_budget_is_enforced_before_execution() -> None:
    calls = tuple(
        {**tool_turn(f"call_{index}").output_items[1]}
        for index in range(1, 3)
    )
    provider = ScriptedProvider(
        [ProviderTurn(response_id="too_many", output_items=calls)]
    )
    tools = FakeTools()
    with pytest.raises(ToolCallLimitError, match="tool-call limit"):
        AssistantClient(
            provider, tools, model="test-model", max_tool_calls=1
        ).answer("Help")
    assert tools.calls == []


def test_empty_provider_response_is_rejected() -> None:
    provider = ScriptedProvider(
        [ProviderTurn(response_id="empty", output_items=())]
    )
    with pytest.raises(ProviderResponseError, match="neither answer"):
        AssistantClient(provider, FakeTools(), model="test-model").answer("Help")


def test_context_is_capped_at_ten_turns() -> None:
    context = [{"role": "user", "content": "x"}] * 21
    provider = ScriptedProvider([final_turn()])
    with pytest.raises(ValueError, match="10-turn limit"):
        AssistantClient(provider, FakeTools(), model="test-model").answer(
            "Help", context=context
        )
    assert provider.requests == []


@pytest.mark.parametrize("model", ["", "   "])
def test_model_is_required(model: str) -> None:
    with pytest.raises(ValueError, match="model"):
        AssistantClient(ScriptedProvider([]), FakeTools(), model=model)


def test_policy_contains_required_grounding_and_safety_boundaries() -> None:
    policy = ASSISTANT_INSTRUCTIONS.lower()
    for phrase in (
        "use only the supplied footcast tools",
        "untrusted data",
        "never invent",
        "not betting",
        "financial advice",
        "api keys",
        "uncertain",
        "generated_at",
        "four tool calls",
    ):
        assert phrase in policy
