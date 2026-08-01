# Phase 9 Typed Assistant Tools

## Purpose and scope

Checkpoint 2 implements the five read-only tools defined by the assistant
benchmark. These are ordinary deterministic Python services. They do not call
an LLM, accept natural language, store conversation history, add an API key, or
change the live dashboard.

The future server-side LLM client will receive the tool catalog, choose one
tool, and submit structured arguments. `AssistantTools.execute` validates those
arguments before calling the same immutable inference and analytics services
already used by FastAPI.

## Tools

| Tool | Strict input | Deterministic source | Principal output evidence |
| --- | --- | --- | --- |
| `get_match_prediction` | Home team, away team, future date | `EloReferenceService` | Three probabilities, ratings, model version, cutoff, warning, timestamp |
| `get_team_form` | Team, limit 1-20 | `AnalyticsService` | Completed matches, summary, date range, aggregation, cutoff, timestamp |
| `compare_teams` | Two teams, limit 1-20 | Both services | Equal-window summaries, sample sizes, ratings, cutoff, timestamp |
| `get_model_explanation` | One approved topic | Model contract and tracked evaluation evidence | Facts, evidence source, model version, test season, limitations, timestamp |
| `get_metric_definition` | One approved term | Versioned FootCast documentation | Definition, interpretation, documentation version, source, timestamp |

All inputs inherit a Pydantic configuration that forbids unknown fields and
strips surrounding whitespace. Limits, dates, topics, and terms are constrained
by the schema before service execution.

## Catalog and dispatcher

`AssistantTools.catalog()` exposes a provider-neutral list of tool names,
descriptions, read-only flags, and JSON Schemas. No provider-specific format is
stored in the tool layer.

`AssistantTools.execute(name, arguments)`:

1. rejects unknown tool names;
2. validates the exact arguments against the selected strict input model;
3. converts prediction and analytics domain errors into a safe
   `AssistantToolError`; and
4. returns one typed, JSON-serializable result.

The tool-call limit, provider retries, rate limiting, and response generation
belong to later server-side assistant checkpoints rather than this deterministic
layer.

## Evidence contract

Every tool result includes a timezone-aware `generated_at` timestamp and a
human-readable source. Data-dependent tools also include the approved data
cutoff, model version, sample/window information, or evidence file as relevant.

The two stateful services must share exactly one data cutoff. Construction fails
if they do not, preventing an answer from mixing prediction state and analytics
from different snapshots.

The model-explanation tool supports only model selection, draw recall, deployed
features, limitations, final-test results, the calibration decision, and the
analytics-versus-prediction boundary. The metric tool supports Elo, macro F1,
log loss, calibration, multiclass Brier score, confusion matrices, and draw
recall. Unsupported topics must not be improvised by this layer.

## Immutability and security boundary

- Tools expose no write, train, update, download, shell, file, or network
  operation.
- Prediction calls do not update Elo ratings.
- Form and comparison calls operate on the immutable in-memory approved
  history.
- No raw data row is returned outside the bounded completed-match form view.
- The 2025-26 holdout is absent because both underlying services already reject
  it.
- There is no credential, provider SDK, prompt, or browser-exposed secret.

## Example without an LLM

```python
from datetime import date

from footcast.analytics.service import AnalyticsService
from footcast.assistant.tools import AssistantTools
from footcast.inference.elo_service import (
    EloReferenceService,
    load_reference_matches,
)

matches = load_reference_matches()
tools = AssistantTools(
    EloReferenceService(matches),
    AnalyticsService(matches),
)
result = tools.execute(
    "get_match_prediction",
    {
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "match_date": date(2026, 8, 15),
    },
)
print(result.model_dump(mode="json"))
```

## Test contract

Focused tests cover the read-only catalog, strict argument validation, domain
failures, probability normalization, prediction-state immutability, form
chronology, sample sizes, every approved explanation and metric, evidence
timestamps, JSON serialization, shared-cutoff enforcement, tracked calibration
facts, and compatibility with all 35 tool-backed benchmark cases.

## Exit condition

Checkpoint 2 is complete when all five tools work deterministically and their
tests pass without an LLM. Checkpoint 3 may then add a configurable server-side
provider client and concise assistant policy while keeping credentials out of
Git and the browser.
