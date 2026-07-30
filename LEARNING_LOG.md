# Learning Log

## Milestone 0: Project initialization

### Concepts

- Soccer matches are time-dependent observations, so evaluation must preserve
  chronology.
- Validation data guides model and feature choices.
- Test data estimates final generalization and should not guide development.
- A final holdout can demonstrate performance after the full system is frozen.
- Feature leakage occurs when a predictor contains information unavailable at
  the prediction time.

### Decisions

- Begin with one league: the English Premier League.
- Predict home win, draw, or away win before kickoff.
- Compare Random Forest against naive, Elo, and logistic-regression baselines.
- Keep raw and generated data out of Git.
- Build reusable logic in the Python package and use notebooks for exploration.

### Questions for the next milestone

- Which Football-Data columns are consistent across every selected season?
- Which match statistics are missing in older files?
- How should team names and promoted teams be normalized?
- What validation rules should stop the pipeline immediately?

## Phase 1: Data ingestion and schema validation

### Concepts

- A download is reproducible only when both its source URL and expected content
  checksum are recorded. A URL alone can later return different bytes.
- Raw-data immutability means validation and canonicalization create new
  artifacts; they never repair the source file in place.
- A canonical schema should contain the stable meaning required by the system,
  not every optional column offered by one source version.
- Schema drift is not automatically corrupt data. Required-column drift must
  stop the pipeline, while optional-column drift should be recorded for review.
- Result validation is relational: `FTR` is valid only when it agrees with the
  full-time home and away goals.

### Decisions

- Pin all eleven Football-Data files by SHA-256 in a JSON manifest.
- Keep the Phase 1 canonical table to seven fields: season, date, teams, goals,
  and normalized outcome.
- Treat all selected seasons as completed Premier League seasons and require
  exactly 380 matches and 20 teams.
- Use July 1 through July 31 of the following year as the declared season
  boundary. This includes the delayed end of the 2019-20 season.
- Report added/removed team names between seasons as promotion/relegation audit
  candidates without claiming an independent source verification.
- Preserve optional match statistics and odds only in raw files for now.

### Observations

- All 4,180 required match records passed validation.
- The source schema has four notable transitions: 65 to 62 columns in 2018-19,
  62 to 106 in 2019-20, 106 to 120 in 2024-25, and 120 to 132 in 2025-26.
- Optional fields contain missing values, especially in recent betting-odds
  columns, but every canonical required field is complete.
- Football-Data uses both two-digit and four-digit years in historical date
  strings, so parsing must deliberately support mixed formats.

### Mistakes and fixes

- The first end-to-end download failed because the local macOS Python did not
  expose a usable CA certificate bundle. The downloader now uses `certifi` and
  still performs normal TLS certificate verification.
- A test originally inserted a decimal into an integer pandas column. Pandas 3
  correctly rejected the test setup before FootCast validation ran; the fixture
  now uses object dtype for deliberately malformed score input.

### Questions for Phase 2

- Which optional match statistics are sufficiently complete and consistently
  defined for responsible exploratory analysis?
- Which team-name aliases, if any, need an explicit long-term normalization
  registry before team histories are built?
- How should exploratory charts clearly distinguish descriptive current-match
  statistics from future leakage-safe model inputs?
