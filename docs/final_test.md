# Phase 5 Checkpoint 2: Frozen Model and Final Test

This checkpoint freezes FootCast v1 before opening the 2024-25 test season.
The purpose is to obtain an unbiased estimate of the complete development
process, not another opportunity to tune the model.

## Frozen contract

The tracked contract is `models/frozen_model_spec.json`, linked to reports and
the local artifact by a SHA-256 hash.

- model version: `footcast-rf-v1`;
- estimator: Random Forest;
- trees: `300`;
- maximum depth: `6`;
- minimum leaf size: `20`;
- class weighting: none;
- calibration: none;
- features: the exact ordered set of 38 Phase 3 pre-match features;
- fit seasons: 2015-16 through 2023-24;
- test season: 2024-25;
- sealed holdout: 2025-26.

Test features are calculated sequentially. A test match may use information
from earlier completed 2024-25 matches, because that information would have
been available before kickoff. It cannot use its own outcome or any later
match.

## Final result

| Model | Accuracy | Macro F1 | Log loss | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Elo | 0.526 | 0.392 | **0.993** | **0.594** | **0.009** |
| Logistic regression | **0.529** | **0.398** | 1.006 | 0.603 | 0.057 |
| Frozen Random Forest | 0.513 | 0.383 | 1.005 | 0.600 | 0.037 |

The frozen forest declined from validation accuracy `0.579` to test accuracy
`0.513`. Macro F1 declined from `0.425` to `0.383`, and log loss worsened from
`0.931` to `1.005`.

It correctly identified 81.3% of home wins and 52.3% of away wins, but zero of
93 draws. It made 33 incorrect predictions with confidence of at least 60%.
Accuracy was `0.612` for large Elo gaps and only `0.365` for close Elo gaps.

## Decision

FootCast v1 is **not recommended for deployment**.

The more complex forest did not provide a stable advantage over Elo or logistic
regression on the untouched test season, and the draw failure remains severe.
The test result is recorded without changing v1. Any future feature, model,
threshold, or calibration change creates a new candidate and cannot reuse
2024-25 as an unseen test.

The generated `models/footcast-rf-v1.joblib` artifact is ignored by Git because
it is environment-specific and reproducible from source. Its checksum and
runtime versions are recorded in `reports/final_test_results.json`.

The 2025-26 holdout remains sealed. It must not be used to repair v1; it may be
used only under a separately approved demonstration or future-model policy.
