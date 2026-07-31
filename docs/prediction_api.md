# Phase 7 Prediction API Contract

## Purpose

The API turns FootCast's historical research into a usable, deterministic
product interface. It serves the transparent Elo reference chosen at the end
of Phase 6. The model can be replaced later without changing the browser-facing
contract.

This is an educational probability demonstration. It is not betting advice and
does not claim deployment-quality accuracy.

## Data and inference boundary

At startup, the service loads only the `train`, `validation`, and previously
seen `test` splits: 3,800 completed matches from 2015-16 through 2024-25. It
replays every result chronologically to reconstruct current Elo ratings and
fits the draw probability as the empirical rate across those matches.

The 2025-26 holdout is not loaded. The service reports its newest completed
match date as `data_cutoff` so clients cannot mistake a stale historical
snapshot for live information.

An incoming fixture contains only home team, away team, and match date. Scoring
does not update ratings. The date must be later than the service cutoff, teams
must be known and different, and extra fields such as a result are rejected.

## Versioned reference model

The tracked `models/elo_reference_spec.json` records:

- model and class order
- approved data splits
- Elo constants
- draw-probability definition
- development evidence
- intended use and limitations

`GET /model/info` also returns the SHA-256 hash of that canonical code contract.
The API model version is `footcast-elo-v2-reference`.

## Endpoints

### `GET /health`

Returns service readiness, model version, data cutoff, and an explicit
`holdout_used: false` field.

### `GET /teams`

Returns the stable list of teams observed in approved completed history.

### `POST /predict`

Request:

```json
{
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "match_date": "2026-08-15"
}
```

Representative response from the approved local data snapshot:

```json
{
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "match_date": "2026-08-15",
  "home_win_probability": 0.5397,
  "draw_probability": 0.2332,
  "away_win_probability": 0.2272,
  "predicted_result": "home_win",
  "home_elo": 1745.83,
  "away_elo": 1660.52,
  "model_version": "footcast-elo-v2-reference",
  "data_cutoff": "2025-05-25",
  "intended_use": "educational probability demonstration",
  "warning": "Not intended for betting or financial decisions."
}
```

### `GET /model/info`

Returns model provenance, constants, evidence, limitations, supported-team
count, completed-match count, and holdout status.

### `GET /analytics/team-form`

Accepts `team` and an optional `limit` from 1 through 20. Returns the latest
completed matches from that team's perspective, including opponent, venue,
goals for and against, W/D/L outcome, points, and an aggregate summary.

### `GET /analytics/compare`

Accepts distinct `home_team` and `away_team` values plus an optional form
limit. Returns each team's recent form and current immutable Elo rating. The
rating difference is home minus away.

### `GET /analytics/head-to-head`

Accepts distinct `team_a` and `team_b` values plus an optional limit. Returns
the latest meetings in descending date order while preserving each fixture's
historical home/away orientation and score.

All responses include an `X-Process-Time-Ms` header for local latency
measurement. FastAPI also exposes interactive OpenAPI documentation at `/docs`.

## Run locally

First reproduce the approved raw files, then start the API:

```bash
python -m footcast.data.pipeline
uvicorn footcast.api.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` or call the endpoints with an HTTP client.

## Test contract

Tests cover chronological replay, immutable prediction state, normalized fixed-
order probabilities, invalid fixtures, unknown teams, cutoff enforcement,
holdout rejection, tracked-spec synchronization, response schemas, extra-field
rejection, endpoint readiness, and latency headers. CI uses injected synthetic
history so tests do not depend on ignored raw files. Analytics tests also cover
team-perspective conversion, latest-N chronology, score orientation, limits,
unknown teams, comparison ratings, and holdout rejection.

## Known limitations

- Draw probability is identical across matchups.
- The service is a historical snapshot, not a live fixture feed.
- Injuries, lineups, transfers, and tactical context are unavailable.
- The team list includes every club observed since 2015-16, not only the latest
  Premier League membership.
- Recent form and head-to-head are descriptive completed-match context, not
  new predictive inputs.
