# FootCast Phase 5 Final Test Report

**Status:** PASSED

The model contract was frozen before the 2024-25 test season was loaded. The
frozen Random Forest uses 38 pre-match features, 300 trees, depth 6, minimum
leaf size 20, no class weighting, and no probability calibration. It was
refitted on 2015-16 through 2023-24, then evaluated on 2024-25 without changing
features, preprocessing, parameters, thresholds, or calibration.

- Model version: `footcast-rf-v1`
- Specification SHA-256: `49913be31b579d051513d712cc2c239e6cf16e414e5e3aa35ecfdb3bec10ca91`
- Fit rows: 3420
- Test rows: 380
- Test season: 2024-25
- Holdout rows loaded: 0

Test-season features remain pre-match. A match may use results from earlier
completed matches in 2024-25, but never its own result or a later result.

## Final 2024-25 model comparison

| Model | Accuracy | Macro F1 | Log loss | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority class | 0.408 | 0.193 | 1.081 | 0.656 | 0.041 |
| Always home | 0.408 | 0.193 | 21.342 | 1.184 | 0.592 |
| Elo | 0.526 | 0.392 | 0.993 | 0.594 | 0.009 |
| Logistic regression | 0.529 | 0.398 | 1.006 | 0.603 | 0.057 |
| Frozen Random Forest | 0.513 | 0.383 | 1.005 | 0.600 | 0.037 |

## Frozen Random Forest class metrics

| Outcome | Precision | Recall |
| --- | ---: | ---: |
| home_win | 0.502 | 0.813 |
| draw | 0.000 | 0.000 |
| away_win | 0.535 | 0.523 |

## Model decision

**Not recommended for deployment.** The frozen forest's test accuracy and
macro F1 fell below its validation results. Logistic regression achieved higher
test accuracy and macro F1, while Elo achieved lower log loss, lower Brier
score, and lower ECE. The frozen forest also selected no draws correctly.

This decision does not replace or retune v1 after seeing the test. It records
that the frozen candidate did not meet the standard for deployment.

## Validation-to-test context

| Metric | 2023-24 validation | 2024-25 test |
| --- | ---: | ---: |
| Accuracy | 0.579 | 0.513 |
| Macro F1 | 0.425 | 0.383 |
| Log loss | 0.931 | 1.005 |
| Brier score | 0.547 | 0.600 |
| ECE | 0.049 | 0.037 |

## Final error-analysis highlights

- Home-win recall: 0.813
- Draw recall: 0.000
- Away-win recall: 0.523
- High-confidence mistakes at or above 60%: 33
- Close Elo-gap accuracy: 0.365
- Large Elo-gap accuracy: 0.612

The test result is the unbiased estimate for the frozen v1 development process.
It must not be used to revise v1 and then reported again as if still unseen.
Any future model changes create a new candidate and require a new evaluation
policy. The 2025-26 holdout remains sealed for a later demonstration.
