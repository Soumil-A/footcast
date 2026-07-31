# Model Card

## Status

All three Phase 6 research checkpoints are complete. `footcast-rf-v1` was
frozen before evaluation on the 2024-25 test season. No post-processing
calibration was retained because both
sigmoid and isotonic methods worsened forward training-period probability
metrics. The frozen forest reached test accuracy `0.513`, macro F1 `0.383`, and
log loss `1.005`. It is not recommended for deployment. Seven v2 rolling
backtests subsequently found that Poisson and Dixon-Coles models had worse mean
log loss than Elo and Random Forest and zero draw recall. They are not
replacement candidates. The 2025-26 holdout remains untouched.

The final two-stage draw-aware checkpoint was also rejected. Its selected
unweighted configuration reached mean rolling log loss `0.996`, Brier score
`0.589`, macro F1 `0.405`, and draw recall `0.016`. Elo is the Phase 7
educational reference model, not an approved system for betting or financial
use.

Phase 7 checkpoint 1 exposes that Elo reference through a versioned API. It
reconstructs ratings from 3,800 approved completed matches, reports a
`2025-05-25` data cutoff, and rejects data from the sealed holdout. The API is a
portfolio demonstration, not evidence that the prediction quality improved.

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

For Phase 6 v2 research, Random Forest and Elo reached mean rolling log loss
`0.975` and `0.976`. Poisson reached `1.017`, and Dixon-Coles reached `1.018`.
The goal models both had zero mean draw recall; logistic regression had the
highest, but still very low, draw recall at `0.024`. These are development
backtests through the already-seen 2024-25 season, not a new final-test claim.

Draw weighting demonstrated a direct tradeoff. Weight `2.0` increased mean
draw recall to `0.339` and macro F1 to `0.451`, but worsened log loss to `1.039`
and Brier score to `0.619`. Optimizing the label decision at that cost would
conflict with FootCast's probability objective.

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
- Independent Poisson team/opponent effects omit injuries, lineups, transfers,
  tactics, and other time-varying context.
- The bounded Dixon-Coles correction changed low-score probabilities but did
  not make draws the most likely class in the rolling evaluation matches.
- Two-stage class weighting can force more draw labels but produces worse
  probability estimates; it is not retained.
- Elo remains unable to identify draws as the most likely class in these
  backtests and is suitable only for an explicitly limited educational demo.
- API predictions use a historical snapshot rather than live injuries,
  lineups, transfers, fixtures, or results.
- The API team list contains all observed clubs since 2015-16, including clubs
  that may not belong to the latest Premier League season.
