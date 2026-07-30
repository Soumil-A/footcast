# FootCast

FootCast is a learning-focused machine learning project for predicting English
Premier League match outcomes before kickoff. The system will estimate the
probability of a home win, draw, or away win using only information that would
have been available before each match.

The project is intentionally being built in stages so that every modeling,
evaluation, and engineering decision is understandable and reproducible.

## Current status

Milestone 0 is complete: the repository structure, initial tests, documentation,
and chronological data-split contract are in place. No data has been downloaded
and no model has been trained yet.

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

The initial dataset will consist of historical Premier League CSV files from
[Football-Data](https://www.football-data.co.uk/englandm.php). Raw datasets are
not committed to Git; download and validation scripts will make acquisition
reproducible in the next milestone.

## Responsible use

FootCast is an educational sports-analytics project. Its predictions are not
financial or betting advice. Soccer outcomes contain substantial irreducible
uncertainty, and model limitations will be documented openly.
