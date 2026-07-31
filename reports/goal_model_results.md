# FootCast Phase 6 Goal-Model Backtest Report

**Status:** PASSED

Phase 6 begins v2 research after the 2024-25 v1 test. Seasons through 2024-25
are now development data. Seven expanding folds train on earlier seasons and
evaluate the next season from 2018-19 through 2024-25. The 2025-26 holdout was
not loaded.

Poisson regression models each team's goals using a team attack effect, an
opponent defense effect, and a home indicator. Dixon-Coles adjusts the four
low-score cells (0-0, 0-1, 1-0, and 1-1), where football outcomes—especially
draws—often depart from independent Poisson assumptions.

## Goal-model settings

### Independent Poisson

| Alpha | Rho | Mean log loss | Mean macro F1 | Mean draw recall |
| ---: | ---: | ---: | ---: | ---: |
| 0.10 | 0.00 **selected** | 1.017 | 0.364 | 0.000 |
| 0.50 | 0.00 | 1.050 | 0.302 | 0.000 |
| 1.00 | 0.00 | 1.059 | 0.275 | 0.000 |

### Dixon-Coles

| Alpha | Rho | Mean log loss | Mean macro F1 | Mean draw recall |
| ---: | ---: | ---: | ---: | ---: |
| 0.10 | -0.05 **selected** | 1.018 | 0.364 | 0.000 |
| 0.10 | -0.10 | 1.020 | 0.364 | 0.000 |
| 0.10 | -0.15 | 1.023 | 0.364 | 0.000 |
| 0.50 | -0.05 | 1.052 | 0.302 | 0.000 |
| 0.50 | -0.10 | 1.054 | 0.302 | 0.000 |
| 0.50 | -0.15 | 1.057 | 0.302 | 0.000 |
| 1.00 | -0.05 | 1.061 | 0.275 | 0.000 |
| 1.00 | -0.10 | 1.064 | 0.275 | 0.000 |
| 1.00 | -0.15 | 1.067 | 0.275 | 0.000 |

Settings are compared as v2 development research on the same rolling folds.
These means are not a new final-test estimate.

## Rolling model comparison

| Model | Accuracy | Macro F1 | Log loss | Brier | Draw recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Elo | 0.547 | 0.401 | 0.976 | 0.580 | 0.000 |
| Logistic regression | 0.535 | 0.409 | 0.996 | 0.589 | 0.024 |
| Random Forest | 0.543 | 0.400 | 0.975 | 0.578 | 0.002 |
| Poisson | 0.515 | 0.364 | 1.017 | 0.609 | 0.000 |
| Dixon-Coles | 0.515 | 0.364 | 1.018 | 0.610 | 0.000 |

The charts show mean metrics, season-to-season log loss, and draw recall. A goal
model is useful only if its probability quality and draw behavior improve
without relying on the sealed holdout. Any v2 selection or feature expansion
must be decided before 2025-26 is opened.

## Conclusion

Neither goal model improved this checkpoint. Random Forest achieved the lowest
mean rolling log loss (`0.975`), with Elo effectively tied at `0.976`; Poisson
and Dixon-Coles reached `1.017` and `1.018`. The selected low-score correction
also did not make draws the most likely class for any evaluation match, so both
goal models had zero draw recall.

This is a negative but useful result. The current goal-rate formulation should
not replace the simpler benchmarks. The next experiment should be specified
before it runs and should target missing pre-match information or the decision
rule for draws—not repeatedly tune these results. The 2025-26 holdout remains
sealed.
