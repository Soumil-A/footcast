"""Lazy OpenAI Responses API adapter for the FootCast assistant."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from footcast.assistant.client import (
    ProviderResponseError,
    ProviderTurn,
    ProviderUsage,
    TransientProviderError,
)


class OpenAIResponsesProvider:
    """Adapt an injected OpenAI client without reading credentials in a browser."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @classmethod
    def from_environment(cls) -> OpenAIResponsesProvider:
        """Create a server-side provider using only ``OPENAI_API_KEY``."""
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise RuntimeError("OPENAI_API_KEY is not configured")
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "Install the FootCast llm extra to enable the OpenAI provider"
            ) from error
        return cls(OpenAI(max_retries=0))

    def create_response(
        self,
        *,
        model: str,
        instructions: str,
        input_items: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        timeout_seconds: float,
    ) -> ProviderTurn:
        try:
            response = self._client.responses.create(
                model=model,
                instructions=instructions,
                input=list(input_items),
                tools=list(tools),
                include=["reasoning.encrypted_content"],
                parallel_tool_calls=False,
                store=False,
                timeout=timeout_seconds,
            )
        except Exception as error:
            if error.__class__.__name__ in {
                "APIConnectionError",
                "APITimeoutError",
                "InternalServerError",
                "RateLimitError",
            }:
                raise TransientProviderError(
                    "OpenAI is temporarily unavailable"
                ) from error
            raise ProviderResponseError("OpenAI request failed") from error

        try:
            output_items = tuple(
                item.model_dump(mode="json") for item in response.output
            )
            usage = response.usage
            return ProviderTurn(
                response_id=response.id,
                output_items=output_items,
                output_text=response.output_text or "",
                usage=ProviderUsage(
                    input_tokens=usage.input_tokens if usage else 0,
                    output_tokens=usage.output_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                ),
            )
        except (AttributeError, TypeError) as error:
            raise ProviderResponseError(
                "OpenAI returned an invalid response"
            ) from error
