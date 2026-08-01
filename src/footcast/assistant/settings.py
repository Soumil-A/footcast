"""Environment configuration for the optional server-side LLM client."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _optional_float(name: str) -> float | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


@dataclass(frozen=True)
class AssistantSettings:
    """Explicit settings; no provider or credential is loaded at import time."""

    model: str
    max_tool_calls: int = 4
    timeout_seconds: float = 10.0
    max_retries: int = 2
    input_usd_per_million_tokens: float | None = None
    output_usd_per_million_tokens: float | None = None

    @classmethod
    def from_environment(cls) -> AssistantSettings:
        model = os.getenv("FOOTCAST_LLM_MODEL", "").strip()
        if not model:
            raise RuntimeError("FOOTCAST_LLM_MODEL is not configured")
        try:
            max_tool_calls = int(os.getenv("FOOTCAST_LLM_MAX_TOOL_CALLS", "4"))
            timeout_seconds = float(os.getenv("FOOTCAST_LLM_TIMEOUT_SECONDS", "10"))
            max_retries = int(os.getenv("FOOTCAST_LLM_MAX_RETRIES", "2"))
        except ValueError as error:
            raise ValueError("Invalid numeric FootCast LLM setting") from error
        if not 1 <= max_tool_calls <= 4:
            raise ValueError("FOOTCAST_LLM_MAX_TOOL_CALLS must be between 1 and 4")
        if timeout_seconds <= 0:
            raise ValueError("FOOTCAST_LLM_TIMEOUT_SECONDS must be positive")
        if not 0 <= max_retries <= 3:
            raise ValueError("FOOTCAST_LLM_MAX_RETRIES must be between 0 and 3")
        return cls(
            model=model,
            max_tool_calls=max_tool_calls,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            input_usd_per_million_tokens=_optional_float(
                "FOOTCAST_LLM_INPUT_USD_PER_MILLION_TOKENS"
            ),
            output_usd_per_million_tokens=_optional_float(
                "FOOTCAST_LLM_OUTPUT_USD_PER_MILLION_TOKENS"
            ),
        )
