# Phase 8 Deployment Contract

## Scope

FootCast packages the Phase 7 product as two independently health-checked
containers and defines their public deployment through a Render Blueprint. It
does not change Elo calculations, API responses, analytics, or dashboard
behavior.

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

## Public deployment

`render.yaml` declares two Docker web services in Render's Virginia region:

- `footcast-api-soumil` builds `docker/api.Dockerfile` and checks `/health`;
- `footcast-dashboard-soumil` builds `docker/dashboard.Dockerfile` and checks
  `/_stcore/health`.

Both use the free plan. The dashboard receives `FOOTCAST_API_URL` from the API
service's Render-generated `RENDER_EXTERNAL_URL`; no deployment hostname or
secret is committed. Auto-deployment waits for GitHub CI checks to pass.

The first deployment requires one account-level action that source code cannot
perform: connect the GitHub repository to a Render workspace.

1. Merge the deployment PR so `render.yaml` is on `main`.
2. Sign in at `https://dashboard.render.com`.
3. Choose **New > Blueprint**, connect `Soumil-A/footcast`, and select `main`.
4. Keep the default Blueprint path, review the two free web services, and apply.
5. Wait until both services report `Live`, then record their public URLs.

Render builds the same Dockerfiles exercised by CI. A free service may need to
wake after inactivity, so the first request can be slower than later requests.

## Public monitoring

The API and dashboard have platform-native health checks. The
`Public deployment monitor` GitHub Actions workflow adds an external end-to-end
check every six hours. It verifies availability plus the scientific serving
contract: 3,800 matches, cutoff `2025-05-25`, and no holdout use.

After Render assigns URLs, enable the scheduled job by setting repository
variables:

```bash
gh variable set FOOTCAST_API_URL --body "https://YOUR-API.onrender.com"
gh variable set FOOTCAST_DASHBOARD_URL --body "https://YOUR-DASHBOARD.onrender.com"
gh workflow run deployment-monitor.yml
```

The same check can be run locally:

```bash
footcast-monitor \
  --api-url "https://YOUR-API.onrender.com" \
  --dashboard-url "https://YOUR-DASHBOARD.onrender.com"
```

If either repository variable is absent, the scheduled job safely skips rather
than reporting a false outage before initial deployment. GitHub records each
run and reports failures through the repository's Actions notifications.

## Operational boundary

This deployment is appropriate for an educational portfolio demo. It has one
instance per service, no uptime service-level agreement, no custom domain, and
no application database. Raw serving history remains immutable inside each API
image. The educational-use and non-betting limitations in the model card remain
unchanged.
