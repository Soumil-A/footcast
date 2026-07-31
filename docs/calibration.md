# Phase 5 Checkpoint 1: Calibration and Error Analysis

This checkpoint determines whether a post-processing calibrator makes the
selected Random Forest probabilities more trustworthy. It also identifies
where the model fails before any final test evaluation.

## Forward-only calibration selection

The selected Phase 4 Random Forest is fixed at 300 trees, depth `6`, minimum
leaf size `20`, and no class weighting.

Five next-season probability blocks are generated inside the training period:
2018-19 through 2022-23. Each block comes from a forest trained only on earlier
seasons.

Calibration methods are then compared sequentially:

1. fit a calibrator on the earliest available out-of-fold season;
2. evaluate it on the next out-of-fold season;
3. add that completed season to calibration training; and
4. repeat through 2022-23.

The comparison covers:

- no calibration;
- multinomial sigmoid calibration on log probabilities;
- one-versus-rest isotonic calibration followed by normalization.

Mean multiclass log loss is the primary metric. Mean multiclass Brier score
breaks an exact tie. The 2023-24 validation season does not choose the method.

## Decision

No calibration was selected:

| Method | Mean log loss | Mean Brier | Mean ECE |
| --- | ---: | ---: | ---: |
| Uncalibrated | **0.994** | **0.590** | **0.034** |
| Sigmoid | 1.003 | 0.596 | 0.052 |
| Isotonic | 1.166 | 0.602 | 0.076 |

Both fitted methods made the forward training-period probability metrics worse.
Keeping the original probabilities is an evidence-based model choice, not a
missing implementation.

On 2023-24 validation, the retained model has log loss `0.931`, Brier score
`0.547`, and expected calibration error `0.049`.

## Error-analysis findings

- Draws remain the largest weakness: 82 matches, zero selected as the
  most-likely class, and log loss `1.440`.
- Home wins are much easier: accuracy `0.811` and log loss `0.688`.
- Large Elo gaps perform best at `0.655` accuracy and `0.846` log loss.
- Medium Elo gaps have the lowest slice accuracy at `0.462`.
- Early-season matches perform better than established-season matches in this
  single validation season. This is an observation, not evidence that early
  matches are generally easier.
- Only one validation row is a complete-history cold start, so its perfect
  result cannot support a general conclusion.
- Twenty-five incorrect predictions have confidence of at least 60%.

Reliability bins and small slices are descriptive and can be noisy. They are
not used to revise the already selected method.

Regenerate this checkpoint with:

```bash
python -m footcast.models.run_calibration
```

The command loads training and validation seasons only. The 2024-25 test and
2025-26 holdout remain sealed. A later checkpoint must explicitly freeze the
pipeline before the one-time test evaluation.
