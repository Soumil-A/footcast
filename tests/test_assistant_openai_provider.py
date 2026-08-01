"""Contract tests for the lazy OpenAI provider and environment settings."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from footcast.assistant.client import ProviderResponseError, TransientProviderError
from footcast.assistant.openai_provider import OpenAIResponsesProvider
from footcast.assistant.settings import AssistantSettings


@dataclass
class DumpableItem:
    value: dict[str, Any]

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return self.value


class FakeResponses:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def test_openai_adapter_uses_responses_api_contract() -> None:
    response = SimpleNamespace(
        id="resp_1",
        output=[DumpableItem({"type": "message"})],
        output_text="Grounded answer",
        usage=SimpleNamespace(input_tokens=12, output_tokens=4, total_tokens=16),
    )
    responses = FakeResponses(response)
    provider = OpenAIResponsesProvider(SimpleNamespace(responses=responses))

    result = provider.create_response(
        model="selected-model",
        instructions="policy",
        input_items=[{"role": "user", "content": "question"}],
        tools=[{"type": "function", "name": "safe_tool"}],
        timeout_seconds=9.0,
    )

    assert result.response_id == "resp_1"
    assert result.output_items == ({"type": "message"},)
    assert result.output_text == "Grounded answer"
    assert result.usage.total_tokens == 16
    assert responses.kwargs["parallel_tool_calls"] is False
    assert responses.kwargs["store"] is False
    assert responses.kwargs["include"] == ["reasoning.encrypted_content"]
    assert responses.kwargs["timeout"] == 9.0
    assert "api_key" not in responses.kwargs


def test_known_transient_openai_error_is_safely_classified() -> None:
    timeout_type = type("APITimeoutError", (Exception,), {})
    provider = OpenAIResponsesProvider(
        SimpleNamespace(responses=FakeResponses(error=timeout_type("private")))
    )
    with pytest.raises(TransientProviderError, match="temporarily unavailable"):
        provider.create_response(
            model="model",
            instructions="policy",
            input_items=[],
            tools=[],
            timeout_seconds=1,
        )


def test_unknown_openai_error_is_redacted() -> None:
    provider = OpenAIResponsesProvider(
        SimpleNamespace(responses=FakeResponses(error=Exception("secret")))
    )
    with pytest.raises(ProviderResponseError, match="request failed") as captured:
        provider.create_response(
            model="model",
            instructions="policy",
            input_items=[],
            tools=[],
            timeout_seconds=1,
        )
    assert "secret" not in str(captured.value)


def test_provider_factory_requires_server_side_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="not configured"):
        OpenAIResponsesProvider.from_environment()


def test_settings_are_loaded_without_reading_the_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FOOTCAST_LLM_MODEL", "chosen-model")
    monkeypatch.setenv("FOOTCAST_LLM_MAX_TOOL_CALLS", "3")
    monkeypatch.setenv("FOOTCAST_LLM_TIMEOUT_SECONDS", "8.5")
    monkeypatch.setenv("FOOTCAST_LLM_MAX_RETRIES", "1")
    monkeypatch.setenv("FOOTCAST_LLM_INPUT_USD_PER_MILLION_TOKENS", "0.25")
    monkeypatch.setenv("FOOTCAST_LLM_OUTPUT_USD_PER_MILLION_TOKENS", "2")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-loaded")

    settings = AssistantSettings.from_environment()

    assert settings.model == "chosen-model"
    assert settings.max_tool_calls == 3
    assert settings.timeout_seconds == 8.5
    assert settings.max_retries == 1
    assert settings.input_usd_per_million_tokens == 0.25
    assert settings.output_usd_per_million_tokens == 2
    assert "key" not in settings.__dict__


def test_settings_require_explicit_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOOTCAST_LLM_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="not configured"):
        AssistantSettings.from_environment()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("FOOTCAST_LLM_MAX_TOOL_CALLS", "5", "between 1 and 4"),
        ("FOOTCAST_LLM_TIMEOUT_SECONDS", "0", "positive"),
        ("FOOTCAST_LLM_MAX_RETRIES", "4", "between 0 and 3"),
        (
            "FOOTCAST_LLM_INPUT_USD_PER_MILLION_TOKENS",
            "-1",
            "non-negative",
        ),
    ],
)
def test_invalid_settings_fail_fast(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    message: str,
) -> None:
    monkeypatch.setenv("FOOTCAST_LLM_MODEL", "model")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        AssistantSettings.from_environment()
