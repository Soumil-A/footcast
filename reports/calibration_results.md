# FootCast Phase 5 Calibration and Error-Analysis Report

**Status:** PASSED

This checkpoint evaluates calibration for the selected Random Forest without
opening the 2024-25 test or 2025-26 holdout seasons. Base-model probabilities
are generated with expanding training-period folds. Each calibration method is
fitted on earlier out-of-fold seasons and evaluated on the next out-of-fold
season.

- Training rows: 3040
- Validation rows: 380
- Out-of-fold calibration rows: 1900
- Calibration-selection evaluation seasons:
  2019-20, 2020-21, 2021-22, 2022-23
- Selected method: uncalibrated
- Test rows loaded: 0
- Holdout rows loaded: 0

## Training-period calibration selection

| Method | Mean log loss | Mean Brier | Mean ECE | Mean macro F1 |
| --- | ---: | ---: | ---: | ---: |
| uncalibrated **selected** | 0.994 | 0.590 | 0.034 | 0.391 |
| sigmoid | 1.003 | 0.596 | 0.052 | 0.394 |
| isotonic | 1.166 | 0.602 | 0.076 | 0.388 |

Log loss is the primary selection metric. Brier score breaks an exact tie.
Expected calibration error (ECE) summarizes the gap between confidence and
accuracy across confidence bins; lower values are better for all three
probability metrics.

## 2023-24 validation

| Model | Accuracy | Macro F1 | Log loss | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uncalibrated Random Forest | 0.579 | 0.425 | 0.931 | 0.547 | 0.049 |
| Selected calibration (uncalibrated) | 0.579 | 0.425 | 0.931 | 0.547 | 0.049 |

Calibration adjusts probabilities rather than creating new match information.
It can improve reliability and log loss while leaving the most-likely class,
accuracy, or draw recall unchanged.

## Errors by actual outcome

| Slice | Rows | Accuracy | Mean confidence | Log loss |
| --- | ---: | ---: | ---: | ---: |
| away_win | 123 | 0.634 | 0.518 | 0.938 |
| home_win | 175 | 0.811 | 0.566 | 0.688 |
| draw | 82 | 0.000 | 0.502 | 1.440 |

## Errors by season timing

Early-season rows are matches where either team had played fewer than five
earlier league matches in that season.

| Slice | Rows | Accuracy | Mean confidence | Log loss |
| --- | ---: | ---: | ---: | ---: |
| early_season | 51 | 0.667 | 0.536 | 0.858 |
| established_season | 329 | 0.565 | 0.537 | 0.942 |

## Errors by prior-history availability

Cold starts contain at least one team with no earlier observed Premier League
history in the dataset.

| Slice | Rows | Accuracy | Mean confidence | Log loss |
| --- | ---: | ---: | ---: | ---: |
| known_history | 379 | 0.578 | 0.536 | 0.932 |
| cold_start | 1 | 1.000 | 0.554 | 0.590 |

## Errors by Elo gap

| Slice | Rows | Accuracy | Mean confidence | Log loss |
| --- | ---: | ---: | ---: | ---: |
| large_over_100 | 203 | 0.655 | 0.607 | 0.846 |
| medium_50_100 | 91 | 0.462 | 0.462 | 1.023 |
| close_0_50 | 86 | 0.523 | 0.449 | 1.033 |

## Highest-confidence mistakes

There were 25 incorrect validation
predictions with at least 60% confidence. The table shows up to 15.

| Date | Match | Actual | Predicted | Confidence |
| --- | --- | --- | --- | ---: |
| 2024-04-14 | Liverpool vs Crystal Palace | away_win | home_win | 0.803 |
| 2024-02-17 | Man City vs Chelsea | draw | home_win | 0.734 |
| 2024-04-14 | Arsenal vs Aston Villa | away_win | home_win | 0.730 |
| 2023-12-16 | Man City vs Crystal Palace | draw | home_win | 0.727 |
| 2023-08-26 | Arsenal vs Fulham | draw | home_win | 0.727 |
| 2023-09-30 | Wolves vs Man City | home_win | away_win | 0.715 |
| 2023-12-03 | Man City vs Tottenham | draw | home_win | 0.710 |
| 2024-03-30 | Chelsea vs Burnley | draw | home_win | 0.696 |
| 2023-12-28 | Arsenal vs West Ham | away_win | home_win | 0.691 |
| 2023-12-17 | Liverpool vs Man United | draw | home_win | 0.685 |
| 2023-11-05 | Luton vs Liverpool | draw | away_win | 0.664 |
| 2024-04-27 | Man United vs Burnley | draw | home_win | 0.660 |
| 2023-12-22 | Aston Villa vs Sheffield United | draw | home_win | 0.651 |
| 2024-02-24 | Man United vs Fulham | away_win | home_win | 0.650 |
| 2023-12-07 | Tottenham vs West Ham | away_win | home_win | 0.646 |

Reliability plots show predicted probability against observed frequency for
each outcome; numbers next to points are bin sample sizes. Slice charts compare
accuracy with mean confidence. Small slices and sparse reliability bins should
not be over-interpreted.

This checkpoint does not freeze a production artifact or authorize test-season
evaluation. Those decisions follow review of these validation-only results.
