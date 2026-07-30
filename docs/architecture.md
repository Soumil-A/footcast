# Architecture

## Initial system boundary

FootCast will separate data acquisition, feature creation, training, evaluation,
and serving so that each stage can be tested independently.

```text
Historical CSV files
        |
        v
Schema and quality validation
        |
        v
Canonical match table
        |
        v
Leakage-safe pre-match features
        |
        +----> Baselines
        |
        +----> Random Forest
                    |
                    v
          Calibration and evaluation
                    |
                    v
             Versioned artifact
                    |
              +-----+------+
              |            |
              v            v
         Prediction API  Dashboard
```

## Design principles

1. Raw data is immutable.
2. Processed data can be regenerated from code.
3. Feature calculations exclude the current and future matches.
4. Model selection uses chronological validation.
5. Test and holdout seasons remain untouched until the model is frozen.
6. Notebooks explore ideas; reusable logic lives in `src/footcast`.
7. Every transformation that could leak information receives a focused test.

## Planned modules

- `footcast.data`: download, schema validation, and cleaning
- `footcast.features`: team histories, rolling form, and Elo
- `footcast.models`: baselines, training, calibration, and inference
- `footcast.evaluation`: metrics, plots, and subgroup analysis

These modules will be introduced only when their milestone begins.
