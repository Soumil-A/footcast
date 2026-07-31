# Phase 4 Checkpoint 2: Random Forest Contract

This checkpoint asks whether a nonlinear tree ensemble improves the
probability forecasts established by the simple baselines.

## Selection without validation-season tuning

The outer development boundary remains unchanged:

- fit and select with 2015-16 through 2022-23;
- compare the selected configuration on 2023-24;
- do not load 2024-25 test or 2025-26 holdout data.

Random Forest settings are selected through five expanding folds inside the
training period. The first fold trains on three seasons and validates on the
fourth. Each later fold adds the completed validation season to training and
validates on the next season. This preserves chronology and gives every
candidate the same sequence of historical tests.

The primary selection metric is mean multiclass log loss because FootCast's
output is a three-way probability forecast. Mean macro F1 breaks an exact
log-loss tie.

## Bounded search

Every candidate uses:

- 300 trees;
- square-root feature sampling;
- random seed `42`;
- median imputation plus missingness indicators fitted independently inside
  each fold.

The grid varies:

- maximum depth: `6`, `12`, or unrestricted;
- minimum samples per leaf: `5` or `20`;
- class weighting: none or balanced subsampling.

This creates 12 candidates and 60 fold fits. It is intentionally small enough
to audit and reproduce. The 2023-24 validation season does not select among
these settings.

## Selected configuration

Training-period cross-validation selected:

- maximum depth `6`;
- minimum leaf size `20`;
- no class weighting;
- mean fold log loss `0.978`;
- mean fold macro F1 `0.398`.

Balanced candidates improved fold macro F1, but their probabilities had worse
mean log loss. Since probability quality is the declared primary objective,
the unweighted configuration was selected.

## Validation interpretation

On 2023-24, Random Forest achieved:

- accuracy `0.579`;
- macro F1 `0.425`;
- log loss `0.931`;
- home-win recall `0.811`;
- draw recall `0.000`;
- away-win recall `0.634`.

It narrowly improves the checkpoint-one probability and macro-F1 results, but
it does not solve draw classification. The improvement is not large enough to
justify declaring a production model or opening the reserved test season.

Impurity feature importance is reported as a model diagnostic. It can favor
continuous or high-cardinality features and divide importance among correlated
predictors, so it is not a causal explanation.

Regenerate the complete experiment with:

```bash
python -m footcast.models.run_random_forest
```

Probability calibration, detailed error analysis, artifact persistence, and
reserved-season evaluation remain later checkpoints.
