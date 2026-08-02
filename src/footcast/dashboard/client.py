"""Small synchronous HTTP client used by the Streamlit dashboard."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class FootCastApiError(RuntimeError):
    """A user-displayable API availability or validation error."""


class FootCastApiClient:
    """Keep the browser-facing application behind FootCast's HTTP contract."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10.0,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._opener = opener

    def health(self) -> dict[str, Any]:
        return self._request("/health")

    def teams(self) -> dict[str, Any]:
        return self._request("/teams")

    def model_info(self) -> dict[str, Any]:
        return self._request("/model/info")

    def portfolio(self) -> dict[str, Any]:
        return self._request("/analytics/portfolio")

    def assistant_status(self) -> dict[str, Any]:
        return self._request("/assistant/status")

    def chat(
        self, message: str, *, session_id: str | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"message": message}
        if session_id is not None:
            payload["session_id"] = session_id
        return self._request(
            "/assistant/chat",
            method="POST",
            payload=payload,
            timeout=35.0,
        )

    def reset_assistant_session(self, session_id: str) -> dict[str, Any]:
        return self._request(
            f"/assistant/sessions/{session_id}", method="DELETE"
        )

    def predict(
        self, home_team: str, away_team: str, match_date: str
    ) -> dict[str, Any]:
        return self._request(
            "/predict",
            method="POST",
            payload={
                "home_team": home_team,
                "away_team": away_team,
                "match_date": match_date,
            },
        )

    def compare(
        self, home_team: str, away_team: str, *, limit: int = 5
    ) -> dict[str, Any]:
        return self._request(
            "/analytics/compare",
            query={
                "home_team": home_team,
                "away_team": away_team,
                "limit": limit,
            },
        )

    def head_to_head(
        self, team_a: str, team_b: str, *, limit: int = 10
    ) -> dict[str, Any]:
        return self._request(
            "/analytics/head-to-head",
            query={"team_a": team_a, "team_b": team_b, "limit": limit},
        )

    def _request(
        self,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        if query:
            url = f"{url}?{urlencode(query)}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with self._opener(
                request,
                timeout=self._timeout if timeout is None else timeout,
            ) as response:
                body = response.read().decode("utf-8")
        except HTTPError as error:
            detail = self._error_detail(error)
            raise FootCastApiError(
                f"FootCast API rejected the request: {detail}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise FootCastApiError(
                "FootCast API is unavailable. Start it and try again."
            ) from error
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as error:
            raise FootCastApiError(
                "FootCast API returned an invalid response."
            ) from error
        if not isinstance(decoded, dict):
            raise FootCastApiError("FootCast API returned an unexpected response.")
        return decoded

    @staticmethod
    def _error_detail(error: HTTPError) -> str:
        try:
            decoded = json.loads(error.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return f"HTTP {error.code}"
        detail = decoded.get("detail", f"HTTP {error.code}")
        return str(detail)
