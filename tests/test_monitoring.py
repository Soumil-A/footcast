"""Tests for public deployment monitoring and Render configuration."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from urllib.error import URLError

import pytest
import yaml

from footcast.monitoring import DeploymentCheckError, check_deployment

PROJECT_ROOT = Path(__file__).parents[1]


class FakeResponse:
    def __init__(self, body: str, status: int = 200) -> None:
        self.status = status
        self._body = BytesIO(body.encode("utf-8"))

    def read(self) -> bytes:
        return self._body.read()

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args) -> None:
        return None


def _successful_opener(request, *, timeout):
    assert timeout == 4.0
    if request.full_url.endswith("/_stcore/health"):
        return FakeResponse("ok")
    if request.full_url.endswith("/health"):
        return FakeResponse(
            json.dumps(
                {
                    "status": "ok",
                    "model_version": "footcast-elo-v2-reference",
                    "data_cutoff": "2025-05-25",
                    "holdout_used": False,
                }
            )
        )
    if request.full_url.endswith("/model/info"):
        return FakeResponse(
            json.dumps(
                {
                    "completed_matches": 3800,
                    "data_cutoff": "2025-05-25",
                    "holdout_seasons_used": [],
                }
            )
        )
    raise AssertionError(request.full_url)


def test_public_monitor_checks_health_and_provenance() -> None:
    report = check_deployment(
        "https://api.example.com/",
        "https://dashboard.example.com",
        timeout=4.0,
        opener=_successful_opener,
    )

    assert report["status"] == "passed"
    assert report["completed_matches"] == 3800
    assert report["data_cutoff"] == "2025-05-25"
    assert report["holdout_used"] is False


def test_public_monitor_rejects_provenance_drift() -> None:
    def drifted_opener(request, *, timeout):
        response = _successful_opener(request, timeout=timeout)
        if request.full_url.endswith("/model/info"):
            return FakeResponse(
                json.dumps(
                    {
                        "completed_matches": 4180,
                        "data_cutoff": "2026-05-24",
                        "holdout_seasons_used": ["2025-26"],
                    }
                )
            )
        return response

    with pytest.raises(DeploymentCheckError, match="match count has drifted"):
        check_deployment(
            "https://api.example.com",
            "https://dashboard.example.com",
            timeout=4.0,
            opener=drifted_opener,
        )


def test_public_monitor_translates_network_failure() -> None:
    def failed_opener(request, *, timeout):
        raise URLError("offline")

    with pytest.raises(DeploymentCheckError, match="Request failed"):
        check_deployment(
            "https://api.example.com",
            "https://dashboard.example.com",
            opener=failed_opener,
        )


def test_render_blueprint_preserves_two_service_boundary() -> None:
    blueprint = yaml.safe_load((PROJECT_ROOT / "render.yaml").read_text())
    services = {service["name"]: service for service in blueprint["services"]}
    api = services["footcast-api-soumil"]
    dashboard = services["footcast-dashboard-soumil"]

    assert api["dockerfilePath"] == "./docker/api.Dockerfile"
    assert api["healthCheckPath"] == "/health"
    assert dashboard["dockerfilePath"] == "./docker/dashboard.Dockerfile"
    assert dashboard["healthCheckPath"] == "/_stcore/health"
    for service in services.values():
        assert service["type"] == "web"
        assert service["runtime"] == "docker"
        assert service["plan"] == "free"
        assert service["branch"] == "main"
        assert service["autoDeployTrigger"] == "checksPass"

    api_url = dashboard["envVars"][0]
    assert api_url["key"] == "FOOTCAST_API_URL"
    assert api_url["fromService"] == {
        "type": "web",
        "name": "footcast-api-soumil",
        "envVarKey": "RENDER_EXTERNAL_URL",
    }


def test_scheduled_monitor_uses_repository_urls() -> None:
    workflow = (PROJECT_ROOT / ".github/workflows/deployment-monitor.yml").read_text()

    assert 'cron: "17 */6 * * *"' in workflow
    assert "vars.FOOTCAST_API_URL" in workflow
    assert "vars.FOOTCAST_DASHBOARD_URL" in workflow
    assert "python -m footcast.monitoring" in workflow
