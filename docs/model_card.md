# Model Card

## Status

Phase 4 checkpoint 1 is complete. Four reference methods have been trained on
2015-16 through 2022-23 and evaluated on 2023-24: majority class, always home,
Elo, and multinomial logistic regression. No model is yet selected, calibrated,
or released for inference. Test and holdout seasons remain untouched.

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
the best log loss at `0.935`. Both had zero draw recall at the selected
most-likely label threshold. These are validation results, not final test
performance.

Brier score, calibration curves, per-class precision, and subgroup evaluation
remain planned for the later calibration and error-analysis checkpoint.

## Known limitations

- Match outcomes contain substantial irreducible uncertainty.
- Promoted teams and early-season matches have limited historical data.
- Historical results may not capture injuries, tactics, transfers, or lineups.
- Team strength and league dynamics change over time.
- Current baselines do not reliably select draws as the most likely outcome.
- Validation has guided comparison, so its scores are not an unbiased final
  generalization estimate.
