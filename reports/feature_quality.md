# FootCast Phase 3 Feature-Quality Report

**Status:** PASSED

- Rows: 3420
- Seasons: 2015-16, 2016-17, 2017-18, 2018-19, 2019-20, 2020-21, 2021-22, 2022-23, 2023-24
- Splits: train, validation
- Pre-match feature columns: 38
- Rows where at least one team has no observed history: 23
- Rows with an unavailable rest-days difference: 23
- Test or holdout rows: 0
- Maximum rolling-window matches: 5
- Unexpected missing values: {}

All rows are retained. Rolling windows contain at most five completed prior
matches. Season counters reset at each new season, while general form and Elo
carry forward from earlier completed matches.

Elo uses an initial rating of 1500,
`K=20`, and a
65-point home adjustment when calculating
the expected result. Each row records the rating before that match; updates
happen only after the result is recorded.

The generated table contains development splits only. The 2024-25 test season
and 2025-26 holdout remain excluded.
