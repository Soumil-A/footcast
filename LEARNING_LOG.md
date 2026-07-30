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

## Phase 2: Exploratory analysis

### Concepts

- Exploratory analysis can legitimately inspect post-match statistics to
  understand the dataset, but that does not make those values valid pre-match
  predictors.
- A final season aggregate answers a retrospective question. A model feature
  must instead be calculated as of each match date using completed history.
- Correlation measures co-movement, not causation, and says nothing by itself
  about when a value becomes available.
- Protecting test and holdout seasons begins before modeling. Looking at their
  outcome patterns while designing features would allow future information to
  influence development decisions.

### Decisions

- Restrict every Phase 2 calculation to training and validation seasons,
  covering 2015-16 through 2023-24.
- Keep exploratory transformations in tested package code and use the notebook
  as a readable guide rather than a second source of business logic.
- Commit the small Phase 2 figures because they communicate the analysis in the
  GitHub repository; continue to regenerate them from code.
- Describe newly appearing teams as promotion/relegation candidates until a
  verified promotion-status source or explicit rule is approved.

### Observations

- Home wins represent 44.91% of development matches, draws 23.19%, and away
  wins 31.90%.
- The 2020-21 season has the lowest home-win rate in the development window at
  37.89%; the charts alone do not establish why.
- Newly appearing teams average 0.931 points per match versus 1.468 for
  continuing teams.
- Shots-on-target difference has the strongest non-goal current-match
  correlation with home-result points (0.543). It is still unavailable before
  kickoff and therefore unsafe in its current-match form.

### Questions for Phase 3

- What default should early-season and newly promoted teams receive before
  enough match history exists?
- Should rolling statistics use only the current season or carry selected
  history across season boundaries?
- How should values be shrunk toward the league average when only one or two
  prior matches exist?

## Phase 3: Leakage-safe pre-match features and Elo

### Concepts

- The safe order is snapshot, predict, observe, then update. Reversing the last
  two steps lets a match influence its own features.
- A rolling window must be defined relative to each team, not relative to rows
  in the league-wide match table.
- Season-to-date state and general historical form have different boundaries.
  Season points reset, while known completed history can carry forward.
- Cold starts are information, not bad rows. History counts let a later model
  distinguish a true zero from a value calculated over no prior matches.

### Decisions

- Use sums for five-match form and production, paired with the number of prior
  matches actually available.
- Carry general form, rest history, expanding averages, and Elo across seasons.
- Reset season matches and season points at each new season.
- Leave first-observation rest days missing rather than inventing a schedule
  assumption; retain every row.
- Initialize unseen teams at Elo `1500`, use `K=20`, and include a fixed
  65-point home adjustment in the expected-score calculation.
- Generate development features only. Test and holdout seasons remain excluded.

### Verification

- The real development table contains 3,420 rows and 38 pre-match feature
  columns.
- Every season contributes all 380 matches; early-history rows are not dropped.
- Twenty-three rows contain at least one completely unseen team, and the same
  23 rows have an unavailable rest-days difference.
- No feature other than explicitly documented rest values is missing.
- The maximum rolling-history count is exactly five.

### Questions for Phase 4

- Which feature subset should each baseline receive so comparisons remain fair?
- How should missing first-match rest values be imputed inside a training
  pipeline without using future information?
- Does Elo alone provide most of the signal, or do recent form and rest improve
  validation performance?
