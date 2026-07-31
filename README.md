# FootCast

FootCast is a learning-focused machine learning project for predicting English
Premier League match outcomes before kickoff. The system will estimate the
probability of a home win, draw, or away win using only information that would
have been available before each match.

The project is intentionally being built in stages so that every modeling,
evaluation, and engineering decision is understandable and reproducible.

## Current status

Phases 1 through 5 and all three Phase 6 research checkpoints are complete.
Football-Data acquisition is checksum-pinned and repeatable, raw files are
immutable, eleven
seasons pass schema and content validation, and the canonical match table can
be regenerated locally. Exploratory analysis uses only the training and
validation seasons and clearly labels current-match statistics as descriptive,
not pre-kickoff features. The feature pipeline creates tested pre-match form,
rest, season-state, and Elo values. Four baselines, including FootCast's first
learned model, establish the validation reference points. A time-aware Random
Forest search provides a modest nonlinear improvement. Forward-only calibration
testing found that its original probabilities outperform sigmoid and isotonic
post-processing. The frozen v1 pipeline has now been evaluated once on the
2024-25 test season. Its performance declined and it is not recommended for
deployment. Phase 6 then evaluated Poisson and Dixon-Coles score models across
seven expanding next-season backtests. They did not outperform Elo or Random
Forest and did not solve draw recognition. A final two-stage draw-aware model
improved draw recall only by damaging probability quality, so the fixed
checkpoint gate rejected it. Elo is the Phase 7 educational reference model.
Phase 7 now serves versioned future-fixture probabilities and deterministic
completed-match analytics through FastAPI. A tested Streamlit dashboard uses
that HTTP contract to present forecasts, Elo ratings, recent form, and
head-to-head history. The 2025-26 holdout remains sealed.

## Planned modeling question

> Given information available before kickoff, what are the probabilities of a
> home win, draw, and away win?

The target will use the historical full-time result:

- `H`: home win
- `D`: draw
- `A`: away win

## Chronological evaluation plan

Football matches are time-dependent, so FootCast will not randomly split
individual matches.

| Dataset | Seasons | Purpose |
| --- | --- | --- |
| Training | 2015-16 through 2022-23 | Fit model parameters |
| Validation | 2023-24 | Select features and tune hyperparameters |
| Test | 2024-25 | Perform one final unbiased evaluation |
| Holdout | 2025-26 | Optional demonstration after the system is frozen |

The test and holdout seasons must not influence feature selection or
hyperparameter tuning.

## Leakage contract

Every prediction feature must be available before kickoff. Rolling and
expanding statistics must exclude the current match, normally by applying a
one-match shift before aggregation.

Examples of allowed inputs:

- previous results and points
- pre-match Elo ratings
- rolling goals, shots, and corners from completed matches
- rest days
- season-to-date statistics calculated before kickoff

Examples of forbidden inputs:

- goals or shots from the match being predicted
- half-time or full-time results
- final league position
- any rolling statistic that includes the current match

See [docs/leakage_review.md](docs/leakage_review.md) for the feature-review
checklist.

## Planned model progression

1. Majority-class and always-home baselines
2. Elo-based baseline
3. Multinomial logistic regression
4. Random Forest
5. Probability calibration and error analysis

Random Forest will be judged against the simpler baselines rather than evaluated
in isolation.

## Repository layout

```text
footcast/
├── data/                 # Raw and reproducibly processed data
├── docs/                 # Architecture, leakage review, and model card
├── models/               # Generated model artifacts (not committed)
├── notebooks/            # Exploration and experiment notebooks
├── reports/figures/      # Generated visualizations
├── src/footcast/         # Reusable application and ML code
├── tests/                # Automated tests
├── LEARNING_LOG.md       # Concepts, decisions, and mistakes
└── pyproject.toml        # Package and tool configuration
```

## Local setup

FootCast requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

## Data

The dataset contains all 4,180 Premier League matches from 2015-16 through
2025-26, downloaded from
[Football-Data](https://www.football-data.co.uk/englandm.php). Football-Data
describes its Premier League files as containing full-time and half-time
results, match statistics, and odds.

The versioned [download manifest](data/download_manifest.json) records each
official URL, local filename, chronological split, expected season bounds,
expected row/team counts, and SHA-256 checksum. The downloader:

- reuses a local raw file only when its checksum matches
- downloads new files to a temporary path before moving them into place
- rejects upstream bytes that do not match the reviewed manifest
- refuses to overwrite a different existing raw file

Raw and processed CSV files remain ignored by Git. They are generated locally;
the manifest, pipeline, tests, and quality reports are committed.

### Canonical match schema

| Column | Rule |
| --- | --- |
| `season` | Manifest season in `YYYY-YY` format |
| `match_date` | Valid ISO date inside the declared season |
| `home_team` | Nonblank, whitespace-checked source team name |
| `away_team` | Nonblank, different from the home team |
| `full_time_home_goals` | Nonnegative integer |
| `full_time_away_goals` | Nonnegative integer |
| `result` | `home_win`, `draw`, or `away_win`, consistent with goals |

Optional source columns are not copied into the Phase 1 canonical table. They
are inventoried in the
[data-quality report](reports/data_quality.md), which documents source schema
drift from 65 columns in early files to 132 columns in 2025-26.

### Reproduce Phase 1

From the repository root, after local setup:

```bash
python -m footcast.data.pipeline
pytest
ruff check .
```

The pipeline creates:

- `data/raw/premier_league_YYYY_YY.csv`: checksum-verified source bytes
- `data/processed/matches.csv`: 4,180 validated canonical match rows
- `reports/data_quality.json`: machine-readable full audit
- `reports/data_quality.md`: reviewable audit summary

Validation stops with one clear error containing every detected problem for a
season. It checks required columns, missing values, dates, season boundaries,
team names, home/away identity, scores, outcomes, score/outcome consistency,
duplicate fixtures, row counts, team counts, and source-schema drift.

## Exploratory analysis

Phase 2 explores 3,420 development matches from 2015-16 through 2023-24. It
deliberately excludes the 2024-25 test season and 2025-26 holdout from every
calculation and chart.

```bash
python -m footcast.exploration
```

The command revalidates its input and regenerates:

- outcome distribution by season
- home-win percentage over time
- team-by-season points-per-match heatmap
- historical home-team versus away-team result heatmap
- current-match numerical correlation heatmap
- source missingness and schema-drift heatmap
- new-team versus continuing-team comparison
- `reports/exploration.md` and `reports/exploration_summary.json`

See the guided notebook at `notebooks/02_exploration.ipynb`. Each visualization
states its question, kickoff-time availability, permitted use, and conclusion
limits. Current-match goals, shots, corners, fouls, cards, and results remain
forbidden as predictors for that same match.

## Pre-match features

Phase 3 transforms completed team history into values available immediately
before each target kickoff:

```bash
python -m footcast.features.build_features
```

The development output contains 38 pre-match feature columns for 3,420 matches.
It includes:

- points, wins, goals, shots, and shots on target from up to five prior matches
- same-role home or away form
- days since the previous observed match
- matches played and points earned earlier in the current season
- expanding historical goals scored and conceded
- pre-match Elo ratings
- home-minus-away matchup differences
- history counts that make cold starts explicit

General history and Elo carry across seasons; season matches and points reset.
Unseen teams start with empty history and Elo `1500`. First-match rest values
remain missing rather than inventing information, and those rows are retained.

See [docs/features.md](docs/features.md) for definitions, required source
columns, availability, leakage risks, and missing-history behavior. The
[feature-quality report](reports/feature_quality.md) summarizes the generated
table. The 2024-25 test season and 2025-26 holdout remain excluded.

## Baseline modeling

Phase 4 checkpoint 1 fits on 3,040 matches from 2015-16 through 2022-23 and
compares once on the 380 matches in 2023-24:

```bash
python -m footcast.models.run_baselines
```

| Model | Accuracy | Macro F1 | Log loss |
| --- | ---: | ---: | ---: |
| Majority class | 0.461 | 0.210 | 1.054 |
| Always home | 0.461 | 0.210 | 19.445 |
| Elo | 0.579 | 0.423 | 0.945 |
| Logistic regression | 0.566 | 0.417 | 0.935 |

Elo leads validation accuracy and macro F1; logistic regression gives the best
three-way probability score. Neither model yet detects draws reliably. The
[baseline contract](docs/baselines.md) explains the four methods, preprocessing,
and metrics. Full results are in the
[baseline report](reports/baseline_results.md), with confusion matrices in
`reports/figures/phase4/`. The command never loads the test or holdout seasons.

## Random Forest

Phase 4 checkpoint 2 evaluates 12 Random Forest configurations across five
expanding, next-season folds entirely within the training period:

```bash
python -m footcast.models.run_random_forest
```

The selected 300-tree forest has depth `6`, a minimum leaf size of `20`, and no
class weighting. On 2023-24 validation it reaches `0.579` accuracy, `0.425`
macro F1, and `0.931` log loss. This narrowly leads the checkpoint-one models,
but its draw recall remains zero.

See the [Random Forest contract](docs/random_forest.md) for the selection
design and the [generated report](reports/random_forest_results.md) for every
candidate, fold, metric, and top training-derived importance. The test and
holdout seasons remain untouched.

## Calibration and error analysis

Phase 5 checkpoint 1 compares no calibration, sigmoid calibration, and isotonic
calibration using only forward out-of-fold predictions from the training
period:

```bash
python -m footcast.models.run_calibration
```

No calibration was selected. The original Random Forest probabilities had the
best training-period mean log loss (`0.994`), Brier score (`0.590`), and
expected calibration error (`0.034`). On 2023-24 validation, the retained model
has log loss `0.931`, Brier score `0.547`, and expected calibration error
`0.049`.

The [calibration contract](docs/calibration.md) explains the forward-only
selection design. The [generated report](reports/calibration_results.md)
documents reliability, outcome, season-timing, prior-history, Elo-gap, and
high-confidence-error diagnostics. Draws remain the dominant weakness.

## Frozen v1 final test

Phase 5 checkpoint 2 freezes the complete model contract, refits it through
2023-24, and evaluates it on 2024-25:

```bash
python -m footcast.models.run_final_test
```

| Model | Accuracy | Macro F1 | Log loss |
| --- | ---: | ---: | ---: |
| Elo | 0.526 | 0.392 | **0.993** |
| Logistic regression | **0.529** | **0.398** | 1.006 |
| Frozen Random Forest | 0.513 | 0.383 | 1.005 |

The frozen forest fell from `0.579` validation accuracy to `0.513` test
accuracy and still had zero draw recall. It did not show a stable advantage over
the simpler models, so v1 is **not recommended for deployment**. The
[frozen-model contract](docs/final_test.md) explains the decision, and the
[final test report](reports/final_test_results.md) contains the complete
evidence. The 2025-26 holdout remains unopened.

## Phase 6 goal-model research

Phase 6 begins v2 research with seasons through 2024-25 treated as development
data and evaluates seven expanding next-season folds:

```bash
python -m footcast.models.run_goal_models
```

| Model | Mean log loss | Mean macro F1 | Mean draw recall |
| --- | ---: | ---: | ---: |
| Elo | 0.976 | 0.401 | 0.000 |
| Logistic regression | 0.996 | **0.409** | **0.024** |
| Random Forest | **0.975** | 0.400 | 0.002 |
| Poisson | 1.017 | 0.364 | 0.000 |
| Dixon-Coles | 1.018 | 0.364 | 0.000 |

The explicit score models did not improve probability quality and still never
selected a draw. This is retained as a documented negative result. See the
[goal-model contract](docs/goal_models.md) and
[generated report](reports/goal_model_results.md). No 2025-26 holdout data is
loaded.

### Draw-aware checkpoint and final decision

Phase 6 checkpoint 2 separates draw detection from home-versus-away prediction:

```bash
python -m footcast.models.run_draw_aware
```

| Draw weight | Mean log loss | Mean macro F1 | Mean draw recall |
| ---: | ---: | ---: | ---: |
| **1.00 selected** | 0.996 | 0.405 | 0.016 |
| 1.25 | 0.999 | 0.415 | 0.046 |
| 1.50 | 1.010 | 0.431 | 0.109 |
| 2.00 | 1.039 | 0.451 | 0.339 |

Increasing draw weight raises draw recall but worsens log loss and Brier score.
Checkpoint 3 therefore **rejects** the two-stage model under the fixed gate.
Elo is selected as the Phase 7 reference for an educational probability demo,
not as a betting-quality system. See the
[draw-aware decision contract](docs/draw_aware.md) and
[generated decision report](reports/draw_aware_results.md).

## Phase 7 product

The first product checkpoint reconstructs immutable Elo state from 3,800
approved completed matches and serves future-fixture probabilities without
loading 2025-26:

```bash
uvicorn footcast.api.main:app --reload
```

Available endpoints are:

- `GET /health`
- `GET /teams`
- `POST /predict`
- `GET /model/info`
- `GET /analytics/team-form`
- `GET /analytics/compare`
- `GET /analytics/head-to-head`

Interactive OpenAPI documentation is available at
`http://127.0.0.1:8000/docs`. Every prediction exposes the model version, data
cutoff, Elo ratings, intended use, and warning. Invalid teams, identical teams,
past dates, and extra post-match fields are rejected. See the
[prediction API contract](docs/prediction_api.md) for the request schema,
representative response, reproduction steps, tests, and limitations.

In a second terminal, run the dashboard:

```bash
streamlit run streamlit_app.py
```

Visit `http://127.0.0.1:8501`. The dashboard calls FastAPI over HTTP and never
imports Elo inference or raw data. It provides fixture controls, three-way
probability bars, recent-form summaries, rating comparison, head-to-head
results, provenance, and limitations. Set `FOOTCAST_API_URL` when the API is
not available at `http://127.0.0.1:8000`. See the
[dashboard contract](docs/dashboard.md) for its architecture and test boundary.

The Phase 7 polish checkpoint adds a responsive dark analytics-console design:
glass panels, team monograms, a probability spectrum, Elo balance indicator,
W/D/L momentum chips, compact match-history cards, and a mobile control drawer.
This is a presentation-only improvement; the API and model outputs are
unchanged.

## Phase 8 container checkpoint

The Phase 7 product is now packaged as separate non-root API and dashboard
images. The API image reproducibly downloads and validates only the 3,800
approved matches through `2025-05-25`; the 2025-26 holdout is explicitly
excluded. The dashboard image contains no raw match data and communicates with
the API only over HTTP.

With Docker Desktop running:

```bash
docker compose up --build --wait
```

Open `http://127.0.0.1:8501` for FootCast or
`http://127.0.0.1:8000/docs` for the API, then stop the stack with
`docker compose down`. Pull requests run lint, all tests, both container builds,
and a live Compose health/provenance smoke test. See the
[container and deployment contract](docs/deployment.md) for the reproducibility,
security, configuration, and CI boundaries.

## Responsible use

FootCast is an educational sports-analytics project. Its predictions are not
financial or betting advice. Soccer outcomes contain substantial irreducible
uncertainty, and model limitations will be documented openly.
