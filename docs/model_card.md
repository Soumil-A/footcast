# Model Card

## Status

Phase 5 is complete. `footcast-rf-v1` was frozen before evaluation on the
2024-25 test season. No post-processing calibration was retained because both
sigmoid and isotonic methods worsened forward training-period probability
metrics. The frozen forest reached test accuracy `0.513`, macro F1 `0.383`, and
log loss `1.005`. It is not recommended for deployment. The 2025-26 holdout
remains untouched.

## Intended use

FootCast is intended for:

- learning supervised machine learning and ML engineering
- exploring historical Premier League patterns
- demonstrating reproducible, time-aware model evaluation

## Out-of-scope use

FootCast is not intended for:

- financial or betting decisions
- claims of certain match outcomes
- evaluating individual players or employees

## Checkpoint evaluation

- accuracy
- macro F1
- per-class recall
- multiclass log loss
- confusion matrix

Elo achieved validation accuracy `0.579`, macro F1 `0.423`, and log loss
`0.945`. Logistic regression achieved accuracy `0.566`, macro F1 `0.417`, and
log loss `0.935`. Random Forest achieved accuracy `0.579`, macro F1 `0.425`,
and the current best log loss at `0.931`. All three had zero draw recall at the
selected most-likely label threshold. These are validation results, not final
test performance.

On the final 2024-25 test, Elo achieved accuracy `0.526` and log loss `0.993`;
logistic regression achieved accuracy `0.529` and macro F1 `0.398`; the frozen
forest achieved accuracy `0.513`, macro F1 `0.383`, and log loss `1.005`.
The forest again had zero draw precision and recall.

Brier score, calibration curves, subgroup evaluation, and per-class precision
are included in the Phase 5 reports.

## Known limitations

- Match outcomes contain substantial irreducible uncertainty.
- Promoted teams and early-season matches have limited historical data.
- Historical results may not capture injuries, tactics, transfers, or lineups.
- Team strength and league dynamics change over time.
- Current baselines do not reliably select draws as the most likely outcome.
- Random Forest improves the validation metrics only narrowly over simpler
  methods.
- Impurity feature importance can be distorted by correlated predictors and is
  not a causal explanation.
- Validation has guided comparison, so its scores are not an unbiased final
  generalization estimate.
- Only one 2023-24 match is a complete-history cold start, so that slice cannot
  support a stable performance claim.
- Reliability-bin estimates are noisy when few matches fall in a bin.
- The final forest underperformed simpler alternatives on important test
  metrics and is not approved for deployment.
- The 2024-25 result cannot be used to tune v1 and then treated as unseen again.
