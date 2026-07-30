# Model Card

## Status

No model has been trained. This document will be completed as the system is
developed.

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

## Planned evaluation

- accuracy
- macro F1
- per-class precision and recall
- multiclass log loss
- Brier score
- calibration curves
- confusion matrix
- performance by season and team-strength group

## Known limitations

- Match outcomes contain substantial irreducible uncertainty.
- Promoted teams and early-season matches have limited historical data.
- Historical results may not capture injuries, tactics, transfers, or lineups.
- Team strength and league dynamics change over time.
