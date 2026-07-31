# Phase 8 Container Contract

## Scope

This checkpoint packages the existing Phase 7 product as two independently
health-checked containers. It does not change Elo calculations, API responses,
analytics, or dashboard behavior, and it is not a public deployment yet.

## Images

- `docker/api.Dockerfile` installs FootCast, prepares the approved serving
  snapshot, and runs Uvicorn on port `8000`.
- `docker/dashboard.Dockerfile` runs Streamlit on port `8501` and contains no
  raw match data or model-serving code path.

Both images use Python 3.11, run as the unprivileged `footcast` user, and have
native health checks. Compose also drops Linux capabilities and prevents
privilege escalation.

## Reproducible serving snapshot

The API image runs this command while it is built:

```bash
python -m footcast.data.serving --raw-dir /app/data/raw
```

The command selects only manifest entries labelled `train`, `validation`, or
`test`. It downloads them with the Phase 1 checksum contract and validates the
combined canonical table before the image can complete. The expected snapshot
contains 3,800 matches through `2025-05-25`. Any incomplete history, schema
failure, checksum mismatch, or attempted `holdout` inclusion fails the build.
The 2025-26 holdout is therefore neither copied from the workstation nor
downloaded into the image.

The build context is intentionally narrow: `.dockerignore` excludes raw data,
processed data, reports, notebooks, caches, local environments, and Git
history. The API image receives only the versioned manifest and model
specification needed to reproduce serving.

## Run locally

Docker Desktop or another Docker Engine with Compose v2 is required.

```bash
docker compose up --build --wait
```

Then open:

- dashboard: `http://127.0.0.1:8501`
- API documentation: `http://127.0.0.1:8000/docs`
- API health: `http://127.0.0.1:8000/health`

The dashboard reaches the API at the internal Compose address
`http://api:8000`; the browser does not need to know that address. Override
host ports without changing the container contract:

```bash
FOOTCAST_API_PORT=8001 FOOTCAST_DASHBOARD_PORT=8502 docker compose up --build --wait
```

Stop the stack with:

```bash
docker compose down
```

## CI gate

Every pull request and push to `main` first runs Ruff and the full pytest suite.
The container job then builds both images, starts the stack, and verifies:

- the API becomes healthy;
- `/model/info` reports 3,800 matches, cutoff `2025-05-25`, and no holdout;
- the Streamlit health endpoint responds successfully.

Docker was not available in the local development environment for this
checkpoint, so GitHub Actions is the authoritative image-build and Compose
smoke-test environment. Python contract tests still inspect the Dockerfiles,
Compose configuration, and build-context exclusions locally.

## Deployment boundary

This checkpoint makes the product portable and continuously tested. A later
Phase 8 checkpoint can publish the images to a hosting platform, configure its
public URL and secrets, and add runtime monitoring. The educational-use and
non-betting limitations in the model card remain unchanged.
