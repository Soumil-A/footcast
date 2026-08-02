# Architecture

## Current system boundary

FootCast will separate data acquisition, feature creation, training, evaluation,
and serving so that each stage can be tested independently.

```text
Versioned download manifest
        |
        v
Checksum-verified raw CSV files
        |
        v
Schema and quality validation
        |
        +----> JSON and Markdown quality reports
        |
        v
Canonical match table (Phase 1 complete)
        |
        +----> Development-only exploration (Phase 2 complete)
        |           |
        |           +----> Figures and interpretation report
        |
        v
Leakage-safe pre-match features (Phase 3 complete)
        |
        +----> Baselines (Phase 4 checkpoint 1 complete)
        |           |
        |           +----> Validation metrics and confusion matrices
        |
        +----> Random Forest (Phase 4 checkpoint 2 complete)
                    |
                    +----> Expanding-season selection
                    |
                    v
          Calibration and evaluation
          (Phase 5 complete)
                    |
                    v
       Frozen v1 contract and local artifact
                    |
                    v
          One-time final test report
                    |
                    v
          Do not deploy frozen v1
                    |
                    v
       Phase 6 v2 rolling research
       Poisson and Dixon-Coles models
                    |
                    v
         Retain benchmarks; goal models
         did not improve the evidence
                    |
                    v
       Two-stage draw-aware checkpoint
                    |
                    v
        Fixed gate rejects candidate;
        Elo is Phase 7 reference model
                    |
                    v
        Prediction API
                    |
                    v
       Versioned Elo inference service
                    |
                    v
       FastAPI health, teams, prediction,
       and model-information endpoints
                    |
                    v
       Read-only analytics endpoints
                    |
                    v
       Streamlit dashboard over HTTP
                    |
                    v
       Separate non-root API and dashboard
       containers with health checks
                    |
                    v
       Compose smoke test in CI
                    |
                    v
       Render Blueprint deploy after CI
                    |
                    v
       Native health checks plus scheduled
       external provenance monitor
```

## Design principles

1. Raw data is immutable.
2. Processed data can be regenerated from code.
3. Feature calculations exclude the current and future matches.
4. Model selection uses chronological validation.
5. Test and holdout seasons remain untouched until the model is frozen.
6. Notebooks explore ideas; reusable logic lives in `src/footcast`.
7. Every transformation that could leak information receives a focused test.

## Modules

- `footcast.data.manifest`: typed, chronological source contract
- `footcast.data.download`: verified atomic downloads and raw-file protection
- `footcast.data.validate`: season-level validation and canonicalization
- `footcast.data.pipeline`: orchestration and quality-report generation
- `footcast.data.serving`: approved-only serving snapshot bootstrap
- `footcast.exploration`: development-only summaries, figures, and report
- `footcast.features.form`: completed team histories and pre-match snapshots
- `footcast.features.elo`: pre-match ratings and post-result updates
- `footcast.features.build_features`: matchup assembly and feature audit
- `footcast.models.baselines`: naive, Elo, and logistic-regression estimators
- `footcast.models.run_baselines`: chronological fitting and report generation
- `footcast.models.random_forest`: deterministic forest and expanding folds
- `footcast.models.run_random_forest`: selection, comparison, and reporting
- `footcast.models.calibration`: forward-only probability calibration
- `footcast.models.run_calibration`: calibration selection and error reporting
- `footcast.models.frozen`: v1 contract, fitting, and artifact serialization
- `footcast.models.run_final_test`: frozen test evaluation and final report
- `footcast.models.goal_models`: Poisson score rates and Dixon-Coles adjustment
- `footcast.models.run_goal_models`: v2 expanding next-season backtests
- `footcast.models.draw_aware`: two-stage draw and decisive classifiers
- `footcast.models.run_draw_aware`: fixed search and promotion decision gate
- `footcast.evaluation.metrics`: fixed-order multiclass model metrics
- `footcast.evaluation.plots`: consistent confusion-matrix visualization
- `footcast.evaluation.calibration_plots`: reliability and slice figures
- `footcast.evaluation.error_analysis`: validation-only diagnostic slices
- `footcast.evaluation.goal_model_plots`: rolling v2 comparison figures
- `footcast.evaluation.draw_aware_plots`: final Phase 6 decision figures
- `footcast.inference.elo_service`: immutable approved-history replay and scoring
- `footcast.analytics.service`: recent form, comparison, and head-to-head views
- `footcast.analytics.portfolio`: audited public final-test evidence
- `footcast.assistant.schemas`: strict provider-neutral tool contracts
- `footcast.assistant.tools`: read-only inference, analytics, and evidence tools
- `footcast.assistant.policy`: versioned grounding and refusal instructions
- `footcast.assistant.client`: bounded provider-neutral tool orchestration
- `footcast.assistant.openai_provider`: lazy server-side Responses API adapter
- `footcast.assistant.settings`: validated environment configuration
- `footcast.api.chat`: typed sessions, provenance responses, and abuse controls
- `footcast.api.main`: validated FastAPI schemas, lifecycle, and endpoints
- `footcast.dashboard.client`: defensive HTTP client and error translation
- `footcast.dashboard.app`: Streamlit presentation with no model imports
- `footcast.monitoring`: deployed health and serving-provenance verification

The API loads approved completed matches once at startup and shares that
in-memory snapshot with inference and analytics. The dashboard is a separate
process and can only reach those capabilities through HTTP.

The container boundary preserves that separation. The API image reproduces its
checksum-verified approved history at build time; the dashboard image contains
no match data and reaches the API by its Compose service name. CI starts both
images and verifies their health and model-data provenance before a change can
merge.

In production, Render builds those same Dockerfiles after the `main` branch CI
checks pass. Its Blueprint injects the API's generated public URL into the
dashboard. Native service health checks control routing, while a scheduled
GitHub workflow verifies both public endpoints and the frozen approved-history
contract every six hours.

## Phase 9 assistant boundary

Checkpoint 1 adds no runtime assistant. The versioned benchmark in
`evals/assistant_questions.jsonl` and the grounding contract in
`docs/assistant_requirements.md` define what later code must satisfy.

The planned runtime remains server-side:

```text
Streamlit chat
      |
      v
FastAPI chat endpoint
      |
      v
Provider-neutral LLM client
      |
      v
Typed read-only FootCast tools
      |
      v
Existing inference, analytics, and documentation contracts
```

The browser will never contain an LLM key or import prediction logic. The LLM
will not calculate match statistics or alter probabilities; every
data-dependent claim must be traceable to a typed deterministic tool result.

Checkpoint 2 implements the typed-tool layer. `AssistantTools` is constructed
from the same prediction and analytics service instances and refuses to start
if their data cutoffs differ. Its catalog contains strict JSON Schemas and
read-only declarations so provider adapters do not control validation or
business logic.

Checkpoint 3 implements the provider-neutral client and optional OpenAI adapter.
The client owns the maximum four-call loop, timeout and retry policy, strict
argument handling, context cap, and usage telemetry. The adapter is lazy and
reads its key only when explicitly constructed on the server. There is still no
FastAPI chat route, browser credential, live provider call, or public chat UI.

Checkpoint 4 adds the FastAPI chat boundary. It stores only ten ephemeral
user/assistant turns under a random session UUID and returns structured evidence
metadata beside the generated answer. Message/body limits, answer limits,
per-IP and per-session throttles, safe error translation, expiration, and reset
are enforced in code. The route remains unavailable unless both model and key
are configured on the server; no assistant state or secret enters Streamlit.

Checkpoint 5 adds the `Ask FootCast` presentation layer. Streamlit stores only
the API session UUID and browser-visible user/assistant messages, sends every
question through `FootCastApiClient`, and renders structured evidence returned
by FastAPI. Suggested prompts, answer-mode labels, safe unavailable behavior,
and reset contain no direct provider or analytics access. The existing Render
deployment remains the single hosting path.
