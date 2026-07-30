# Architecture

## Current system boundary

FootCast will separate data acquisition, feature creation, training, evaluation,
and serving so that each stage can be tested independently.

```text
Versioned download manifest
        |
        v
Checksum-verified raw CSV files
        |
        v
Schema and quality validation
        |
        +----> JSON and Markdown quality reports
        |
        v
Canonical match table (Phase 1 complete)
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

## Modules

- `footcast.data.manifest`: typed, chronological source contract
- `footcast.data.download`: verified atomic downloads and raw-file protection
- `footcast.data.validate`: season-level validation and canonicalization
- `footcast.data.pipeline`: orchestration and quality-report generation

Planned later:

- `footcast.features`: team histories, rolling form, and Elo
- `footcast.models`: baselines, training, calibration, and inference
- `footcast.evaluation`: metrics, plots, and subgroup analysis

These modules will be introduced only when their milestone begins.
