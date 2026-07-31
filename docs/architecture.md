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
        +----> Development-only exploration (Phase 2 complete)
        |           |
        |           +----> Figures and interpretation report
        |
        v
Leakage-safe pre-match features (Phase 3 complete)
        |
        +----> Baselines (Phase 4 checkpoint 1 complete)
        |           |
        |           +----> Validation metrics and confusion matrices
        |
        +----> Random Forest (Phase 4 checkpoint 2 complete)
                    |
                    +----> Expanding-season selection
                    |
                    v
          Calibration and evaluation
          (Phase 5 complete)
                    |
                    v
       Frozen v1 contract and local artifact
                    |
                    v
          One-time final test report
                    |
                    v
          Do not deploy frozen v1
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
- `footcast.exploration`: development-only summaries, figures, and report
- `footcast.features.form`: completed team histories and pre-match snapshots
- `footcast.features.elo`: pre-match ratings and post-result updates
- `footcast.features.build_features`: matchup assembly and feature audit
- `footcast.models.baselines`: naive, Elo, and logistic-regression estimators
- `footcast.models.run_baselines`: chronological fitting and report generation
- `footcast.models.random_forest`: deterministic forest and expanding folds
- `footcast.models.run_random_forest`: selection, comparison, and reporting
- `footcast.models.calibration`: forward-only probability calibration
- `footcast.models.run_calibration`: calibration selection and error reporting
- `footcast.models.frozen`: v1 contract, fitting, and artifact serialization
- `footcast.models.run_final_test`: frozen test evaluation and final report
- `footcast.evaluation.metrics`: fixed-order multiclass model metrics
- `footcast.evaluation.plots`: consistent confusion-matrix visualization
- `footcast.evaluation.calibration_plots`: reliability and slice figures
- `footcast.evaluation.error_analysis`: validation-only diagnostic slices

Planned later only after a new approved development policy:

- v2 research, inference API, and dashboard work

These modules will be introduced only when their milestone begins.
