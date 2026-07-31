"""End-to-end checks for a deployed FootCast API and dashboard."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

EXPECTED_MATCHES = 3_800
EXPECTED_CUTOFF = "2025-05-25"


class DeploymentCheckError(RuntimeError):
    """The public product is unavailable or violates its serving contract."""


def _public_base_url(value: str, label: str) -> str:
    normalized = value.rstrip("/") + "/"
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DeploymentCheckError(f"{label} must be an absolute HTTP(S) URL")
    return normalized


def _read(
    base_url: str,
    path: str,
    *,
    timeout: float,
    opener: Callable[..., Any],
) -> tuple[int, str]:
    request = Request(
        urljoin(base_url, path.lstrip("/")),
        headers={"Accept": "application/json", "User-Agent": "footcast-monitor/1"},
    )
    try:
        with opener(request, timeout=timeout) as response:
            return int(response.status), response.read().decode("utf-8")
    except (OSError, TimeoutError) as error:
        raise DeploymentCheckError(f"Request failed for {request.full_url}") from error


def _json_object(body: str, endpoint: str) -> dict[str, Any]:
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as error:
        raise DeploymentCheckError(f"{endpoint} returned invalid JSON") from error
    if not isinstance(decoded, dict):
        raise DeploymentCheckError(f"{endpoint} returned an unexpected payload")
    return decoded


def check_deployment(
    api_url: str,
    dashboard_url: str,
    *,
    timeout: float = 30.0,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """Verify availability and the approved production data contract."""
    api_base = _public_base_url(api_url, "API URL")
    dashboard_base = _public_base_url(dashboard_url, "Dashboard URL")

    health_status, health_body = _read(
        api_base, "/health", timeout=timeout, opener=opener
    )
    health = _json_object(health_body, "/health")
    if health_status != 200 or health.get("status") != "ok":
        raise DeploymentCheckError("API health check did not report ready")
    if health.get("holdout_used") is not False:
        raise DeploymentCheckError("API health check reports holdout use")

    info_status, info_body = _read(
        api_base, "/model/info", timeout=timeout, opener=opener
    )
    info = _json_object(info_body, "/model/info")
    if info_status != 200:
        raise DeploymentCheckError("Model information endpoint is unavailable")
    if info.get("completed_matches") != EXPECTED_MATCHES:
        raise DeploymentCheckError("Production match count has drifted")
    if info.get("data_cutoff") != EXPECTED_CUTOFF:
        raise DeploymentCheckError("Production data cutoff has drifted")
    if info.get("holdout_seasons_used") != []:
        raise DeploymentCheckError("Production model information includes holdout")

    dashboard_status, dashboard_body = _read(
        dashboard_base, "/_stcore/health", timeout=timeout, opener=opener
    )
    if dashboard_status != 200 or dashboard_body.strip().lower() != "ok":
        raise DeploymentCheckError("Dashboard health check did not report ready")

    return {
        "status": "passed",
        "api_url": api_base.rstrip("/"),
        "dashboard_url": dashboard_base.rstrip("/"),
        "model_version": health.get("model_version"),
        "completed_matches": info["completed_matches"],
        "data_cutoff": info["data_cutoff"],
        "holdout_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-url", default=os.environ.get("FOOTCAST_PUBLIC_API_URL")
    )
    parser.add_argument(
        "--dashboard-url", default=os.environ.get("FOOTCAST_PUBLIC_DASHBOARD_URL")
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if not args.api_url or not args.dashboard_url:
        parser.error("public API and dashboard URLs are required")
    report = check_deployment(args.api_url, args.dashboard_url, timeout=args.timeout)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
