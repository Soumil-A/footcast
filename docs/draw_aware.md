# Phase 6 Draw-Aware Decision Contract

## Question

Can a model designed explicitly around draws improve draw recognition without
sacrificing the probability quality established by Elo and Random Forest?

This is a bounded v2 development experiment. Seasons through 2024-25 are
already seen development data. The 2025-26 holdout remains sealed.

## Two-stage formulation

The first logistic classifier estimates draw versus non-draw. The second is
trained only on decisive matches and estimates home win versus away win. The
probabilities are combined as:

```text
P(draw) = draw model
P(home win) = [1 - P(draw)] * P(home | decisive)
P(away win) = [1 - P(draw)] * [1 - P(home | decisive)]
```

Both classifiers use the existing 38 leakage-safe pre-match features. Median
imputation, missingness indicators, scaling, and estimation are fitted inside
each training fold. Draw weights `1.0`, `1.25`, `1.5`, and `2.0` were fixed
before running the real data.

## Evaluation and selection

The same seven expanding next-season folds used in checkpoint 1 evaluate
2018-19 through 2024-25. Candidate selection minimizes mean multiclass log
loss, using macro F1 and then lower draw weight only for exact ties.

Checkpoint 3 promotes the selected model only if every predeclared gate passes:

- mean log loss at most `0.970`
- mean multiclass Brier score at most `0.578`
- mean macro F1 at least `0.400`
- mean draw recall at least `0.100`

## Evidence

| Draw weight | Log loss | Brier | Macro F1 | Draw recall |
| ---: | ---: | ---: | ---: | ---: |
| **1.00 selected** | 0.996 | 0.589 | 0.405 | 0.016 |
| 1.25 | 0.999 | 0.591 | 0.415 | 0.046 |
| 1.50 | 1.010 | 0.598 | 0.431 | 0.109 |
| 2.00 | 1.039 | 0.619 | 0.451 | 0.339 |

Weighting makes the tradeoff visible: higher weights recognize more draws but
produce worse probabilities. Because probability quality is FootCast's primary
output, the unweighted candidate is selected and then evaluated by the fixed
gate. It passes macro F1 but fails log loss, Brier score, and draw recall.

## Decision

The two-stage model is **rejected**. Elo becomes the Phase 7 reference because
it is simple, effectively tied with Random Forest on rolling log loss, and beat
the frozen forest on the original 2024-25 test log loss. It will be exposed
only as an educational probability demonstration with explicit limitations.

This decision ends the bounded Phase 6 v2 model search. Phase 7 should build a
deterministic prediction API before adding a dashboard or LLM assistant.

## Reproduce

```bash
python -m footcast.models.run_draw_aware
```

The command regenerates JSON and Markdown evidence plus the model comparison,
season comparison, and draw-weight tradeoff figures.
