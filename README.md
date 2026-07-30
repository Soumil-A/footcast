# FootCast

FootCast is a learning-focused machine learning project for predicting English
Premier League match outcomes before kickoff. The system will estimate the
probability of a home win, draw, or away win using only information that would
have been available before each match.

The project is intentionally being built in stages so that every modeling,
evaluation, and engineering decision is understandable and reproducible.

## Current status

Phases 1 through 3 are complete. Football-Data acquisition is checksum-pinned and
repeatable, raw files are immutable, eleven seasons pass schema and content
validation, and the canonical match table can be regenerated locally.
Exploratory analysis uses only the training and validation seasons and clearly
labels current-match statistics as descriptive, not pre-kickoff features. The
feature pipeline now creates tested pre-match form, rest, season-state, and Elo
values. No model has been trained.

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

## Responsible use

FootCast is an educational sports-analytics project. Its predictions are not
financial or betting advice. Soccer outcomes contain substantial irreducible
uncertainty, and model limitations will be documented openly.
