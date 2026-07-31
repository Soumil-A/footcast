FROM python:3.11-slim-bookworm

ARG APP_UID=10001
ARG APP_GID=10001

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FOOTCAST_API_URL=http://api:8000

WORKDIR /app

RUN groupadd --gid "${APP_GID}" footcast \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home footcast

COPY pyproject.toml README.md streamlit_app.py ./
COPY src ./src
COPY .streamlit ./.streamlit

RUN python -m pip install --no-cache-dir . \
    && chown -R footcast:footcast /app

USER footcast

EXPOSE 8501

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=4 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)" || exit 1

CMD ["sh", "-c", "streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true"]
