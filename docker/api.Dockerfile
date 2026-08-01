FROM python:3.11-slim-bookworm

ARG APP_UID=10001
ARG APP_GID=10001

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FOOTCAST_PROJECT_ROOT=/app \
    FOOTCAST_MANIFEST=/app/data/download_manifest.json \
    FOOTCAST_RAW_DIR=/app/data/raw

WORKDIR /app

RUN groupadd --gid "${APP_GID}" footcast \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home footcast

COPY pyproject.toml README.md ./
COPY src ./src
COPY data/download_manifest.json ./data/download_manifest.json
COPY models/elo_reference_spec.json ./models/elo_reference_spec.json

RUN python -m pip install --no-cache-dir ".[llm]" \
    && python -m footcast.data.serving --raw-dir /app/data/raw \
    && chown -R footcast:footcast /app

USER footcast

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=4 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["sh", "-c", "uvicorn footcast.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
