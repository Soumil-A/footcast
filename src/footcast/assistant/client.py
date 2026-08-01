"""Bounded, provider-neutral orchestration for the FootCast assistant."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from footcast.assistant.policy import ASSISTANT_INSTRUCTIONS
from footcast.assistant.schemas import AssistantToolDescriptor, AssistantToolResult

LOGGER = logging.getLogger(__name__)


class AssistantClientError(RuntimeError):
    """Safe base error for assistant orchestration failures."""


class ProviderUnavailableError(AssistantClientError):
    """A provider request failed after the bounded retry policy."""


class TransientProviderError(ProviderUnavailableError):
    """A provider timeout, rate limit, connection, or server failure."""


class ProviderResponseError(AssistantClientError):
    """The provider returned an invalid or incomplete response."""


class ToolCallLimitError(AssistantClientError):
    """The provider attempted to exceed the configured tool-call budget."""


class ToolCallProtocolError(AssistantClientError):
    """A provider emitted an unknown tool or invalid arguments."""


@dataclass(frozen=True)
class ProviderUsage:
    """Provider-reported token usage for one response."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class FunctionCall:
    """One normalized function call emitted by a provider."""

    call_id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class ProviderTurn:
    """Provider-neutral response while preserving opaque output items."""

    response_id: str
    output_items: tuple[dict[str, Any], ...]
    output_text: str = ""
    usage: ProviderUsage = field(default_factory=ProviderUsage)

    @property
    def function_calls(self) -> tuple[FunctionCall, ...]:
        calls: list[FunctionCall] = []
        for item in self.output_items:
            if item.get("type") != "function_call":
                continue
            call_id = item.get("call_id")
            name = item.get("name")
            arguments = item.get("arguments")
            if not all(isinstance(value, str) and value for value in (call_id, name)):
                raise ProviderResponseError(
                    "Provider returned an invalid function call"
                )
            if not isinstance(arguments, str):
                raise ProviderResponseError("Provider returned invalid tool arguments")
            calls.append(FunctionCall(call_id, name, arguments))
        return tuple(calls)


class AssistantProvider(Protocol):
    """Minimal provider boundary required by the orchestrator."""

    def create_response(
        self,
        *,
        model: str,
        instructions: str,
        input_items: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        timeout_seconds: float,
    ) -> ProviderTurn: ...


class AssistantToolRegistry(Protocol):
    """Read-only tool surface consumed by the orchestrator."""

    def catalog(self) -> list[AssistantToolDescriptor]: ...

    def execute(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> AssistantToolResult: ...


@dataclass(frozen=True)
class AssistantEvidence:
    """Display-safe provenance extracted from one typed tool result."""

    tool_name: str
    answer_mode: str
    generated_at: str
    source: str
    data_cutoff: str | None = None
    model_version: str | None = None
    test_season: str | None = None
    window: int | None = None
    sample_size: dict[str, Any] | None = None
    documentation_version: str | None = None


@dataclass(frozen=True)
class AssistantRun:
    """A completed answer plus operational metadata safe for server logs."""

    answer: str
    model: str
    provider_responses: int
    tool_calls: int
    tool_names: tuple[str, ...]
    evidence: tuple[AssistantEvidence, ...]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float | None
    latency_ms: int


def _strict_tool_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a Pydantic object schema to the Responses strict-tool subset."""
    normalized = json.loads(json.dumps(schema))
    properties = normalized.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("Assistant tool input schema must define object properties")
    normalized["additionalProperties"] = False
    normalized["required"] = list(properties)
    for definition in properties.values():
        if isinstance(definition, dict):
            definition.pop("default", None)
    return normalized


def responses_tool_catalog(
    descriptors: Sequence[AssistantToolDescriptor],
) -> list[dict[str, Any]]:
    """Adapt provider-neutral descriptors to Responses function tools."""
    return [
        {
            "type": "function",
            "name": descriptor.name,
            "description": descriptor.description,
            "parameters": _strict_tool_schema(descriptor.input_schema),
            "strict": True,
        }
        for descriptor in descriptors
    ]


class AssistantClient:
    """Execute a bounded model/tool loop without exposing provider credentials."""

    def __init__(
        self,
        provider: AssistantProvider,
        tools: AssistantToolRegistry,
        *,
        model: str,
        instructions: str = ASSISTANT_INSTRUCTIONS,
        max_tool_calls: int = 4,
        max_context_items: int = 20,
        max_answer_chars: int = 12_000,
        timeout_seconds: float = 10.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.25,
        input_usd_per_million_tokens: float | None = None,
        output_usd_per_million_tokens: float | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not model.strip():
            raise ValueError("Assistant model must not be empty")
        if not 1 <= max_tool_calls <= 4:
            raise ValueError("max_tool_calls must be between 1 and 4")
        if not 0 <= max_context_items <= 20:
            raise ValueError("max_context_items must be between 0 and 20")
        if not 1 <= max_answer_chars <= 12_000:
            raise ValueError("max_answer_chars must be between 1 and 12000")
        if timeout_seconds <= 0 or max_retries < 0 or retry_backoff_seconds < 0:
            raise ValueError("Timeout and retry settings must be non-negative")
        prices = (input_usd_per_million_tokens, output_usd_per_million_tokens)
        if any(price is not None and price < 0 for price in prices):
            raise ValueError("Token prices must be non-negative")
        self._provider = provider
        self._tools = tools
        self._model = model.strip()
        self._instructions = instructions
        self._max_tool_calls = max_tool_calls
        self._max_context_items = max_context_items
        self._max_answer_chars = max_answer_chars
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._input_price = input_usd_per_million_tokens
        self._output_price = output_usd_per_million_tokens
        self._clock = clock
        self._sleep = sleep
        self._provider_tools = responses_tool_catalog(tools.catalog())
        self._known_tools = {tool["name"] for tool in self._provider_tools}

    @property
    def model(self) -> str:
        """Configured model identifier for status and observability."""
        return self._model

    def answer(
        self,
        question: str,
        *,
        context: Sequence[Mapping[str, Any]] = (),
    ) -> AssistantRun:
        """Answer one question through a bounded, auditable tool loop."""
        if not question.strip():
            raise ValueError("Assistant question must not be empty")
        if len(context) > self._max_context_items:
            raise ValueError("Assistant context exceeds the 10-turn limit")
        input_items = [dict(item) for item in context]
        input_items.append({"role": "user", "content": question.strip()})
        started = self._clock()
        responses = 0
        tool_names: list[str] = []
        evidence: list[AssistantEvidence] = []
        usage = ProviderUsage()

        while True:
            turn = self._request(input_items)
            responses += 1
            usage = ProviderUsage(
                input_tokens=usage.input_tokens + turn.usage.input_tokens,
                output_tokens=usage.output_tokens + turn.usage.output_tokens,
                total_tokens=usage.total_tokens + turn.usage.total_tokens,
            )
            input_items.extend(turn.output_items)
            calls = turn.function_calls
            if not calls:
                answer = turn.output_text.strip()
                if not answer:
                    raise ProviderResponseError(
                        "Provider returned neither answer text nor a tool call"
                    )
                if len(answer) > self._max_answer_chars:
                    raise ProviderResponseError(
                        "Provider answer exceeded the configured size limit"
                    )
                elapsed_ms = round((self._clock() - started) * 1000)
                result = AssistantRun(
                    answer=answer,
                    model=self._model,
                    provider_responses=responses,
                    tool_calls=len(tool_names),
                    tool_names=tuple(tool_names),
                    evidence=tuple(evidence),
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                    estimated_cost_usd=self._estimate_cost(usage),
                    latency_ms=elapsed_ms,
                )
                self._log_completion(result)
                return result

            if len(tool_names) + len(calls) > self._max_tool_calls:
                raise ToolCallLimitError("Assistant exceeded the tool-call limit")
            for call in calls:
                if call.name not in self._known_tools:
                    raise ToolCallProtocolError(
                        f"Provider requested unknown tool: {call.name!r}"
                    )
                arguments = self._parse_arguments(call)
                try:
                    tool_result = self._tools.execute(call.name, arguments)
                except (TypeError, ValueError) as error:
                    raise ToolCallProtocolError(
                        f"Provider supplied invalid arguments for {call.name}"
                    ) from error
                tool_names.append(call.name)
                evidence.append(self._extract_evidence(tool_result))
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": tool_result.model_dump_json(),
                    }
                )

    @staticmethod
    def _extract_evidence(tool_result: AssistantToolResult) -> AssistantEvidence:
        payload = tool_result.model_dump(mode="json")
        sample_size = payload.get("sample_size")
        return AssistantEvidence(
            tool_name=str(payload["tool_name"]),
            answer_mode=str(payload["answer_mode"]),
            generated_at=str(payload["generated_at"]),
            source=str(payload["source"]),
            data_cutoff=payload.get("data_cutoff"),
            model_version=payload.get("model_version"),
            test_season=payload.get("test_season"),
            window=payload.get("window"),
            sample_size=sample_size if isinstance(sample_size, dict) else None,
            documentation_version=payload.get("documentation_version"),
        )

    def _request(self, input_items: Sequence[Mapping[str, Any]]) -> ProviderTurn:
        for attempt in range(self._max_retries + 1):
            try:
                return self._provider.create_response(
                    model=self._model,
                    instructions=self._instructions,
                    input_items=input_items,
                    tools=self._provider_tools,
                    timeout_seconds=self._timeout_seconds,
                )
            except TransientProviderError:
                if attempt == self._max_retries:
                    raise ProviderUnavailableError(
                        "Assistant provider is temporarily unavailable"
                    ) from None
                self._sleep(self._retry_backoff_seconds * (2**attempt))
        raise AssertionError("unreachable")

    @staticmethod
    def _parse_arguments(call: FunctionCall) -> dict[str, Any]:
        try:
            arguments = json.loads(call.arguments_json)
        except json.JSONDecodeError as error:
            raise ToolCallProtocolError(
                f"Provider returned malformed arguments for {call.name}"
            ) from error
        if not isinstance(arguments, dict):
            raise ToolCallProtocolError(
                f"Provider returned non-object arguments for {call.name}"
            )
        return arguments

    def _estimate_cost(self, usage: ProviderUsage) -> float | None:
        if self._input_price is None or self._output_price is None:
            return None
        return round(
            (
                usage.input_tokens * self._input_price
                + usage.output_tokens * self._output_price
            )
            / 1_000_000,
            8,
        )

    @staticmethod
    def _log_completion(result: AssistantRun) -> None:
        LOGGER.info(
            "assistant_request_complete",
            extra={
                "assistant_model": result.model,
                "assistant_provider_responses": result.provider_responses,
                "assistant_tool_calls": result.tool_calls,
                "assistant_tool_names": result.tool_names,
                "assistant_input_tokens": result.input_tokens,
                "assistant_output_tokens": result.output_tokens,
                "assistant_total_tokens": result.total_tokens,
                "assistant_estimated_cost_usd": result.estimated_cost_usd,
                "assistant_latency_ms": result.latency_ms,
            },
        )
