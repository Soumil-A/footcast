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

## Phase 4 Checkpoint 1: Baselines

### Concepts

- A baseline is a reference point, not merely a weak model. It tells us whether
  added complexity earns a measurable improvement.
- Accuracy can conceal total failure on a minority outcome. Macro F1 and
  per-class recall force the comparison to account for draws and away wins.
- A classifier's label and its probabilities answer different questions. Elo
  led label metrics here, while logistic regression produced the better log
  loss.
- Imputation and scaling learn parameters. They belong inside the training
  pipeline so validation values cannot influence training medians or means.

### Decisions

- Fit every checkpoint model on 2015-16 through 2022-23 and evaluate on
  2023-24; do not load 2024-25 or 2025-26.
- Give logistic regression all 38 numeric pre-match features as the first
  transparent learned model.
- Give Elo only its two rating snapshots and estimate its draw probability from
  training outcomes.
- Report accuracy, macro F1, per-class recall, log loss, and fixed-order
  confusion matrices for every model.
- Keep this checkpoint reproducible from source rather than saving a model
  artifact before model selection and calibration.

### Observations

- Elo reached `0.579` validation accuracy and `0.423` macro F1, leading the
  checkpoint's label metrics.
- Logistic regression reached `0.566` accuracy and `0.417` macro F1, but its
  `0.935` log loss narrowly beat Elo's `0.945`.
- The majority and always-home rules made identical labels, yet always-home's
  certainty increased log loss from `1.054` to `19.445`.
- Neither Elo nor logistic regression correctly selected a validation draw.
  Draw handling is therefore a clear later error-analysis and calibration
  target, not a reason to open the test season early.

### Questions for the next Phase 4 checkpoint

- Can Random Forest improve macro F1 or log loss without overfitting the single
  validation season?
- Which hyperparameters should be explored with time-aware cross-validation
  inside the training period?
- Do class weighting or calibrated probabilities improve draw behavior without
  damaging overall probability quality?

## Phase 4 Checkpoint 2: Random Forest

### Concepts

- Hyperparameter tuning needs its own evaluation boundary. Reusing the outer
  validation season for every choice would gradually overfit decisions to that
  season.
- Expanding season folds imitate the real forecasting direction: learn from
  completed seasons, then evaluate on the next unseen season.
- Model selection must name a primary metric before results are viewed.
  FootCast prioritizes log loss because the product promises probabilities,
  while macro F1 remains an important label-quality diagnostic.
- Feature importance explains how this fitted forest splits its data; it does
  not establish that a feature causes match outcomes.

### Decisions

- Use five expanding folds entirely within the eight training seasons.
- Search 12 combinations of maximum depth, minimum leaf size, and class
  weighting with 300 trees and a fixed random seed.
- Fit median imputation and missingness indicators independently inside every
  fold.
- Select lowest mean log loss, using mean macro F1 only to break an exact tie.
- Compare the selected forest once on 2023-24 and continue to exclude test and
  holdout seasons.

### Observations

- Cross-validation selected depth `6`, minimum leaf size `20`, and no class
  weighting, with mean log loss `0.978`.
- Balanced class weighting improved mean fold macro F1 as high as `0.474`, but
  worsened probability quality to roughly `0.999` log loss.
- Random Forest reached `0.579` validation accuracy, `0.425` macro F1, and
  `0.931` log loss. This is a narrow improvement over Elo and logistic
  regression, not a decisive leap.
- Draw recall remained `0.000`. More model complexity did not by itself solve
  the minority-outcome problem.
- Elo difference was the largest impurity importance, supporting the earlier
  observation that stable team-strength information carries substantial
  signal. Correlated Elo fields mean these values must not be read causally.

### Questions for the next checkpoint

- Are the Random Forest probabilities systematically over- or under-confident?
- Can calibration improve probability quality without changing the ranking
  signal?
- Which seasons, teams, cold starts, and confidence ranges contain the largest
  errors?

## Phase 5 Checkpoint 1: Calibration and error analysis

### Concepts

- Calibration is a learned transformation and therefore needs its own
  chronological training boundary.
- Selecting no transformation is a valid result when every fitted calibrator
  worsens forward probability metrics.
- Log loss evaluates the probability assigned to the true outcome, while
  Brier score measures squared error across all three outcomes and expected
  calibration error compares confidence with observed accuracy.
- Slice analysis can reveal failure modes, but small groups and single seasons
  must not be generalized beyond their evidence.

### Decisions

- Generate five out-of-fold probability blocks with forests trained only on
  earlier seasons.
- Compare identity, multinomial sigmoid, and one-versus-rest isotonic
  calibration by fitting on earlier out-of-fold seasons and evaluating on the
  next block.
- Select mean log loss first and Brier score only as an exact-tie breaker.
- Analyze 2023-24 by outcome, season timing, prior-history availability, Elo
  gap, confidence, and highest-confidence mistakes.
- Keep 2024-25 test and 2025-26 holdout data unloaded.

### Observations

- Uncalibrated probabilities won the forward comparison with mean log loss
  `0.994`, Brier score `0.590`, and expected calibration error `0.034`.
- Sigmoid calibration slightly worsened mean log loss to `1.003`; isotonic
  calibration worsened it to `1.166`.
- The retained model's 2023-24 metrics remain log loss `0.931`, Brier `0.547`,
  and expected calibration error `0.049`.
- Draws have validation log loss `1.440` and zero classification accuracy,
  making them the clearest unresolved failure mode.
- Large Elo-gap matches perform better than close or medium-gap matches.
- Twenty-five incorrect predictions carry at least 60% confidence.
- The sole cold-start validation row was correct, but a sample of one cannot
  establish cold-start quality.

### Questions for the next checkpoint

- Is the pipeline sufficiently frozen to authorize the one-time 2024-25 test?
- Which exact code, feature list, constants, and no-calibration decision belong
  in the versioned model artifact?
- What acceptance criteria should stop deployment even if aggregate test
  metrics look acceptable?
