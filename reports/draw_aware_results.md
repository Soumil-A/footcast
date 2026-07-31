# FootCast Phase 6 Draw-Aware Decision Report

**Pipeline status:** PASSED
**Checkpoint 3 decision:** REJECT

Checkpoint 2 tests a two-stage logistic model. The first classifier estimates
draw versus non-draw; the second estimates home versus away conditional on a
decisive match. Their probabilities are combined into the fixed order home win,
draw, and away win. Four draw weights were fixed before the results were run.

All seven backtests train on earlier seasons and evaluate the next season from
2018-19 through 2024-25. The 2025-26 holdout was not loaded.

## Checkpoint 2 candidate search

| Draw weight | Log loss | Brier | Macro F1 | Draw recall |
| ---: | ---: | ---: | ---: | ---: |
| 1.00 **selected** | 0.996 | 0.589 | 0.405 | 0.016 |
| 1.25 | 0.999 | 0.591 | 0.415 | 0.046 |
| 1.50 | 1.010 | 0.598 | 0.431 | 0.109 |
| 2.00 | 1.039 | 0.619 | 0.451 | 0.339 |

## Reference comparison

| Model | Log loss | Brier | Macro F1 | Draw recall |
| --- | ---: | ---: | ---: | ---: |
| Elo | 0.976 | 0.580 | 0.401 | 0.000 |
| Random Forest | 0.975 | 0.578 | 0.400 | 0.002 |
| Two-stage draw-aware | 0.996 | 0.589 | 0.405 | 0.016 |

## Checkpoint 3 fixed acceptance gate

| Criterion | Result |
| --- | --- |
| `mean_log_loss_at_most_0.970` | FAIL |
| `mean_brier_at_most_0.578` | FAIL |
| `mean_macro_f1_at_least_0.400` | PASS |
| `mean_draw_recall_at_least_0.100` | FAIL |

The gate requires every criterion to pass. The decision is **REJECT**.
The Phase 7 reference model is **Elo**, used
only as an educational probability demonstration with documented limitations.

Increasing draw weight may improve draw recall while worsening probability
quality. The candidate table and tradeoff chart preserve that evidence rather
than treating one label metric as sufficient. This decision does not open or
make any claim about the sealed 2025-26 holdout.
