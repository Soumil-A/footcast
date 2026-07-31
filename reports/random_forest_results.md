# FootCast Phase 4 Random Forest Report

**Status:** PASSED

Checkpoint 2 selects Random Forest settings using expanding chronological folds
inside the training period, refits the selected configuration on every training
row, and compares it once on 2023-24 validation. The 2024-25 test and 2025-26
holdout seasons were neither loaded nor inspected.

- Training rows: 3040
- Validation rows: 380
- Numeric pre-match features: 38
- Candidate configurations: 12
- Expanding folds per candidate: 5
- Trees per forest: 300
- Selection rule: lowest mean log loss, then highest macro F1
- Selected configuration: depth=6, leaf=20, class weight=none

## Training-period model selection

| Candidate | Mean log loss | Mean macro F1 |
| --- | ---: | ---: |
| depth=6, leaf=5, class weight=none | 0.981 | 0.399 |
| depth=6, leaf=5, class weight=balanced_subsample | 0.998 | 0.467 |
| depth=6, leaf=20, class weight=none **selected** | 0.978 | 0.398 |
| depth=6, leaf=20, class weight=balanced_subsample | 0.999 | 0.474 |
| depth=12, leaf=5, class weight=none | 0.985 | 0.407 |
| depth=12, leaf=5, class weight=balanced_subsample | 0.994 | 0.445 |
| depth=12, leaf=20, class weight=none | 0.980 | 0.402 |
| depth=12, leaf=20, class weight=balanced_subsample | 0.999 | 0.466 |
| depth=None, leaf=5, class weight=none | 0.986 | 0.411 |
| depth=None, leaf=5, class weight=balanced_subsample | 0.993 | 0.438 |
| depth=None, leaf=20, class weight=none | 0.980 | 0.402 |
| depth=None, leaf=20, class weight=balanced_subsample | 0.999 | 0.465 |

The folds always train on earlier seasons and validate on the immediately
following season. No fold trains on data later than its validation season.

| Training seasons | Validation season | Accuracy | Macro F1 | Log loss |
| --- | --- | ---: | ---: | ---: |
| 2015-16, 2016-17, 2017-18 | 2018-19 | 0.587 | 0.430 | 0.916 |
| 2015-16, 2016-17, 2017-18, 2018-19 | 2019-20 | 0.524 | 0.386 | 0.977 |
| 2015-16, 2016-17, 2017-18, 2018-19, 2019-20 | 2020-21 | 0.508 | 0.375 | 1.045 |
| 2015-16, 2016-17, 2017-18, 2018-19, 2019-20, 2020-21 | 2021-22 | 0.553 | 0.414 | 0.968 |
| 2015-16, 2016-17, 2017-18, 2018-19, 2019-20, 2020-21, 2021-22 | 2022-23 | 0.537 | 0.388 | 0.985 |

## 2023-24 validation comparison

| Model | Accuracy | Macro F1 | Log loss |
| --- | ---: | ---: | ---: |
| Majority class | 0.461 | 0.210 | 1.054 |
| Always home | 0.461 | 0.210 | 19.445 |
| Elo | 0.579 | 0.423 | 0.945 |
| Logistic regression | 0.566 | 0.417 | 0.935 |
| Random Forest | 0.579 | 0.425 | 0.931 |

## Per-class recall

| Model | Home win | Draw | Away win |
| --- | ---: | ---: | ---: |
| Majority class | 1.000 | 0.000 | 0.000 |
| Always home | 1.000 | 0.000 | 0.000 |
| Elo | 0.834 | 0.000 | 0.602 |
| Logistic regression | 0.777 | 0.000 | 0.642 |
| Random Forest | 0.811 | 0.000 | 0.634 |

## Largest training-derived feature importances

| Feature | Impurity importance |
| --- | ---: |
| elo_difference | 0.1802 |
| home_elo | 0.0919 |
| home_expanding_goals_for_mean | 0.0710 |
| shots_on_target_difference | 0.0668 |
| away_elo | 0.0638 |
| away_expanding_goals_for_mean | 0.0561 |
| away_expanding_goals_against_mean | 0.0510 |
| home_expanding_goals_against_mean | 0.0391 |
| away_shots_last_5 | 0.0350 |
| form_points_difference | 0.0298 |
| home_shots_on_target_last_5 | 0.0267 |
| away_shots_on_target_last_5 | 0.0267 |
| home_shots_last_5 | 0.0225 |
| home_season_points | 0.0217 |
| goals_scored_difference | 0.0206 |

Impurity importance describes how often a fitted forest uses a feature to
reduce node impurity. Correlated features can divide or distort importance, so
this is a diagnostic rather than a causal explanation.

The model-comparison chart keeps label metrics separate from log loss because
higher is better for accuracy and macro F1 while lower is better for log loss.
The confusion matrix shows the selected forest's class errors. Probability
calibration, detailed subgroup analysis, and reserved-season evaluation remain
outside this checkpoint.
