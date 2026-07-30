# Phase 4 Checkpoint 1: Baseline Contract

This checkpoint establishes simple, reproducible reference points before any
Random Forest work or hyperparameter tuning.

## Frozen split

| Role | Seasons | Rows | Permitted use |
| --- | --- | ---: | --- |
| Training | 2015-16 through 2022-23 | 3,040 | Fit model parameters and preprocessing |
| Validation | 2023-24 | 380 | Compare checkpoint models |
| Test | 2024-25 | 0 loaded | Reserved for final evaluation |
| Holdout | 2025-26 | 0 loaded | Reserved until the system is frozen |

The runner requests only the `train` and `validation` splits from the validated
loader. A second guard rejects any feature table containing a `test` or
`holdout` row.

## Compared methods

### Majority class

The predicted label is the most frequent training outcome. Its probability
vector is the empirical training class distribution, so validation outcomes do
not influence the prior.

### Always home

Every match is predicted as a home win with probability one. This intentionally
simple rule gives the same labels as the majority baseline for this training
window, but its overconfident probabilities produce much worse log loss.

### Elo

The two pre-match Elo ratings produce an expected home score using the Phase 3
65-point home adjustment. The training draw frequency is reserved as the draw
probability, and the remaining probability is divided between home and away
using the expected score. No result from the target match enters its ratings.

### Multinomial logistic regression

This is FootCast's first learned machine-learning model. It receives all 38
numeric, leakage-safe pre-match features. A single scikit-learn pipeline:

1. fills missing rest values with training medians and adds missingness
   indicators;
2. scales the resulting numeric values using training statistics; and
3. fits multinomial probabilities with logistic regression.

The pipeline is fitted once on training rows and only transformed on validation
rows. Dates, season/split labels, team names, and the result target are not
predictors.

## Metrics

- **Accuracy** is the fraction of correct labels.
- **Macro F1** calculates F1 for each outcome and gives all three equal weight.
- **Per-class recall** shows the fraction of each actual outcome detected.
- **Multiclass log loss** evaluates the probability assigned to the actual
  outcome; lower is better and confident mistakes are strongly penalized.
- **Confusion matrices** expose which outcomes are confused with one another.

Accuracy alone is misleading when classes are uneven. In this validation
season, predicting home wins exclusively is correct 46.1% of the time but has
zero recall for draws and away wins.

## Checkpoint result

Elo has the strongest validation accuracy and macro F1. Logistic regression has
slightly lower accuracy and macro F1 but the best log loss, meaning its full
probability estimates are marginally better. Neither approach recognizes draws
reliably as the most likely class. This is a development finding to investigate
later, not a reason to inspect the reserved test season.

Regenerate the comparison with:

```bash
python -m footcast.models.run_baselines
```

The command writes the Markdown and JSON reports plus the four confusion
matrices. It does not save a production model or load reserved seasons.
