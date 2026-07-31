"""Static contracts for container security and service wiring."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


def _read(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def test_api_image_bootstraps_only_approved_serving_data() -> None:
    dockerfile = _read("docker/api.Dockerfile")

    assert "python -m footcast.data.serving" in dockerfile
    assert "FOOTCAST_MANIFEST=/app/data/download_manifest.json" in dockerfile
    assert "COPY data/raw" not in dockerfile
    assert "USER footcast" in dockerfile
    assert "HEALTHCHECK" in dockerfile


def test_dashboard_image_contains_no_data_and_calls_api_by_environment() -> None:
    dockerfile = _read("docker/dashboard.Dockerfile")

    assert "COPY data" not in dockerfile
    assert "FOOTCAST_API_URL=http://api:8000" in dockerfile
    assert "USER footcast" in dockerfile
    assert "/_stcore/health" in dockerfile


def test_compose_waits_for_api_and_uses_internal_service_address() -> None:
    compose = _read("compose.yaml")

    assert "condition: service_healthy" in compose
    assert "FOOTCAST_API_URL: http://api:8000" in compose
    assert compose.count("no-new-privileges:true") == 2
    assert compose.count("cap_drop:") == 2


def test_docker_context_cannot_include_raw_or_processed_data() -> None:
    dockerignore = _read(".dockerignore")

    assert dockerignore.splitlines()[0] == "**"
    assert "!data/download_manifest.json" in dockerignore
    assert "!data/raw" not in dockerignore
    assert "!data/processed" not in dockerignore
