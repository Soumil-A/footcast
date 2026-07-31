# Phase 6 Goal-Model Research Contract

## Question

Can an explicit football score model improve three-way match probabilities or
draw recognition relative to FootCast's Elo, logistic-regression, and Random
Forest benchmarks?

This is v2 development research, not another use of the v1 test. After the v1
decision was finalized, 2024-25 became previously seen development data. The
2025-26 holdout remains sealed.

## Models

The independent Poisson model creates two training observations per match: one
for each team's goals. Its predictors are the scoring team, opponent, and a
home indicator. Regularized Poisson regression therefore learns team attack,
opponent defence, and home-advantage effects from earlier matches.

For a fixture, the model predicts expected home and away goals. Independent
Poisson probabilities for scores from 0-0 through 10-10 are summed into home
win, draw, and away win probabilities, then normalized in that fixed order.

Dixon-Coles uses the same fitted goal rates and adjusts the probabilities of
0-0, 0-1, 1-0, and 1-1. Those low-score outcomes are where independent Poisson
assumptions are often least realistic. This bounded checkpoint compares rho
values `-0.05`, `-0.10`, and `-0.15` with regularization strengths `0.1`, `0.5`,
and `1.0`.

## Evaluation policy

Seven expanding next-season folds are used:

| Training seasons begin | Evaluation season |
| --- | --- |
| 2015-16 through 2017-18 | 2018-19 |
| 2015-16 through 2018-19 | 2019-20 |
| 2015-16 through 2019-20 | 2020-21 |
| 2015-16 through 2020-21 | 2021-22 |
| 2015-16 through 2021-22 | 2022-23 |
| 2015-16 through 2022-23 | 2023-24 |
| 2015-16 through 2023-24 | 2024-25 |

Configuration selection minimizes mean multiclass log loss, with mean macro F1
as an exact-tie breaker. Every benchmark is refitted on the same earlier-season
rows and evaluated on the same next-season rows. Accuracy, macro F1, log loss,
multiclass Brier score, and draw recall are reported.

The pipeline rejects loaded holdout rows. It neither reads nor evaluates
2025-26, and these rolling means are not described as a new final-test result.

## Result

The selected independent Poisson setting is alpha `0.10`; the selected
Dixon-Coles setting is alpha `0.10`, rho `-0.05`.

| Model | Mean log loss | Mean macro F1 | Mean draw recall |
| --- | ---: | ---: | ---: |
| Elo | 0.976 | 0.401 | 0.000 |
| Logistic regression | 0.996 | **0.409** | **0.024** |
| Random Forest | **0.975** | 0.400 | 0.002 |
| Poisson | 1.017 | 0.364 | 0.000 |
| Dixon-Coles | 1.018 | 0.364 | 0.000 |

The goal models did not improve probability quality and did not solve draw
classification. Dixon-Coles changed low-score probability mass, but the draw
probability still never became the largest of the three outcome probabilities
in these evaluation rows. Random Forest has the lowest mean log loss by only
`0.001` over Elo, so that difference should not be treated as decisive.

## Reproduce

```bash
python -m footcast.models.run_goal_models
```

This generates the machine-readable and Markdown reports plus three figures in
`reports/figures/phase6/`. The negative result is retained so future work does
not repeat the same experiment or promote complexity without evidence.
