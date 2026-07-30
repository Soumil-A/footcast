# FootCast Phase 4 Baseline Report

**Status:** PASSED

This first modeling checkpoint compares four deliberately simple methods on the
same chronological boundary. Models fit on 2015-16 through 2022-23 and are
evaluated on 2023-24. The 2024-25 test and 2025-26 holdout seasons were neither
loaded nor inspected.

- Training rows: 3040
- Validation rows: 380
- Leakage-safe numeric features: 38
- Class order: home_win, draw, away_win

## Overall validation metrics

| Model | Accuracy | Macro F1 | Log loss |
| --- | ---: | ---: | ---: |
| Majority class | 0.461 | 0.210 | 1.054 |
| Always home | 0.461 | 0.210 | 19.445 |
| Elo | 0.579 | 0.423 | 0.945 |
| Logistic regression | 0.566 | 0.417 | 0.935 |

## Per-class recall

| Model | Home win | Draw | Away win |
| --- | ---: | ---: | ---: |
| Majority class | 1.000 | 0.000 | 0.000 |
| Always home | 1.000 | 0.000 | 0.000 |
| Elo | 0.834 | 0.000 | 0.602 |
| Logistic regression | 0.777 | 0.000 | 0.642 |

Accuracy is not sufficient on its own: a model can score reasonably by mostly
choosing the common home-win class while failing to recognize draws or away
wins. Macro F1 gives each outcome equal influence, per-class recall exposes
which outcomes are missed, and log loss evaluates the quality and confidence
of all three probabilities. Lower log loss is better; higher accuracy, macro
F1, and recall are better.

The majority baseline uses training outcome frequencies as probabilities. The
always-home baseline is intentionally deterministic, so confident mistakes
receive a severe log-loss penalty. Elo uses only the two pre-match ratings and
the training draw rate. Logistic regression uses all leakage-safe numeric
features; its median imputer and scaler are fitted on training rows only.

See `reports/figures/phase4/baseline_confusion_matrices.png` for counts by
actual and predicted class. These validation results are a development
checkpoint, not a final estimate of future performance.
