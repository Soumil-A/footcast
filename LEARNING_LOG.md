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

## Phase 5 Checkpoint 2: Frozen model and final test

### Concepts

- A test set evaluates the entire development process, including feature,
  model, hyperparameter, and calibration choices.
- Freezing first prevents test results from quietly becoming another tuning
  signal.
- A lower calibration error does not imply better discrimination or accuracy.
  Probability reliability and outcome separation measure different qualities.
- Simpler baselines remain valuable at the final checkpoint because complexity
  must demonstrate a stable benefit, not merely a validation advantage.

### Decisions

- Freeze 38 ordered features, the selected Random Forest parameters, training
  median imputation, and the no-calibration decision in a tracked JSON contract.
- Refit on training plus validation seasons, then evaluate 2024-25 once without
  changing the contract.
- Generate a local ignored `joblib` artifact and record its checksum, contract
  hash, size, and runtime versions in the final JSON report.
- Compare the frozen forest with the already-defined naive, Elo, and logistic
  baselines on the same test season.
- Do not load the 2025-26 holdout.

### Final observations

- Frozen Random Forest accuracy declined from `0.579` validation to `0.513`
  test; macro F1 declined from `0.425` to `0.383`.
- Test log loss worsened from `0.931` to `1.005`, while Elo achieved `0.993`.
- Logistic regression achieved the best test accuracy (`0.529`) and macro F1
  (`0.398`) among the learned/reference strength models.
- The forest recalled 81.3% of home wins, 52.3% of away wins, and zero draws.
- Close Elo-gap matches had only `0.365` accuracy, compared with `0.612` for
  large gaps.
- Thirty-three incorrect predictions carried confidence of at least 60%.

### Outcome

FootCast v1 is not recommended for deployment. This is a successful learning
result: the untouched test exposed that the small validation advantage did not
generalize. Any v2 work must treat 2024-25 as previously seen and define a new
evaluation policy before development begins.

## Phase 6: Poisson and Dixon-Coles goal models

### Concepts

- Modeling home and away goal counts provides a football-specific route to
  three-way outcome probabilities by summing a score matrix.
- Independent Poisson assumptions can misrepresent low-scoring dependence;
  Dixon-Coles adjusts the four lowest score cells without changing the fitted
  attack and defence rates.
- A previously opened test season may enter later development research, but it
  can never become unseen again. A new sealed boundary is required.
- A plausible domain model is still only a hypothesis. It must outperform
  simpler references on the same chronological folds.

### Decisions

- Treat 2015-16 through 2024-25 as v2 development data and preserve 2025-26 as
  the untouched holdout.
- Use seven expanding next-season folds beginning with evaluation on 2018-19.
- Compare three Poisson regularization strengths and nine Dixon-Coles
  alpha/rho combinations, selecting mean log loss first.
- Refit Elo, logistic regression, and the Phase 4 Random Forest on every fold
  so all five model families receive the same information.
- Stop after the bounded experiment and record a negative result instead of
  tuning repeatedly against the rolling backtests.

### Observations

- Random Forest had the lowest mean log loss at `0.975`, effectively tied with
  Elo at `0.976`; logistic regression reached `0.996`.
- Independent Poisson reached mean log loss `1.017`; Dixon-Coles reached
  `1.018`. Neither improved the benchmarks.
- Poisson and Dixon-Coles both had zero draw recall. The low-score correction
  changed probabilities but never made draw the highest-probability outcome.
- Logistic regression had the best macro F1 (`0.409`) and the highest draw
  recall, but `0.024` remains too small to call the issue solved.

### Outcome and next question

The goal models are not promoted. Phase 6 establishes that a football-specific
formulation alone is insufficient with these inputs. Before opening 2025-26,
the next checkpoint should choose a bounded hypothesis around richer
pre-match information or a draw-aware decision strategy and define its success
criterion in advance.

## Phase 6 Checkpoint 2: Draw-aware two-stage model

### Concepts

- A hierarchical classifier can decompose a multiclass question into draw
  versus non-draw and home versus away conditional on a decisive match.
- Class weighting changes the fitted decision surface and often changes the
  meaning of predicted probabilities; higher minority-class recall is not
  automatically better probability estimation.
- A three-way product can be reconstructed coherently by reserving the draw
  probability and dividing the remaining mass between home and away.

### Decisions

- Use the existing 38 leakage-safe features and seven expanding folds.
- Test only four predeclared draw weights: `1.0`, `1.25`, `1.5`, and `2.0`.
- Select mean log loss first because FootCast presents probabilities.
- Keep preprocessing and both classifiers inside each chronological fold.
- Continue to exclude 2025-26.

### Observations

- Draw recall rose monotonically from `0.016` at weight `1.0` to `0.339` at
  weight `2.0`.
- Macro F1 also rose from `0.405` to `0.451`.
- The same weighting worsened log loss from `0.996` to `1.039` and Brier score
  from `0.589` to `0.619`.
- The unweighted model won the predeclared probability-first selection.

## Phase 6 Checkpoint 3: Fixed model decision

### Decision gate

- Mean log loss at most `0.970`: failed at `0.996`.
- Mean Brier score at most `0.578`: failed at `0.589`.
- Mean macro F1 at least `0.400`: passed at `0.405`.
- Mean draw recall at least `0.100`: failed at `0.016`.

### Outcome

Reject the two-stage model and end the bounded v2 model search. Elo becomes the
Phase 7 reference because it is simpler, effectively tied with Random Forest
on rolling probability quality, and performed better than the forest on the
original final-test log loss. It remains an educational reference with serious
draw limitations, not a deployment-quality forecasting claim. The 2025-26
holdout remains sealed.

## Phase 7 Checkpoint 1: Prediction API

### Concepts

- Training code consumes completed outcomes; inference code must accept an
  unfinished fixture without inventing goals, shots, or a result.
- Replaying approved completed matches creates a point-in-time model state.
  Scoring a hypothetical fixture must not mutate that state.
- A stable HTTP contract allows the model to change later without rewriting
  the dashboard.
- Model provenance belongs in the product response: version, specification
  hash, data cutoff, intended use, and limitations are part of correctness.

### Decisions

- Serve the Phase 6 Elo reference as `footcast-elo-v2-reference`.
- Load only train, validation, and previously seen test splits, totaling 3,800
  completed matches; continue excluding 2025-26.
- Require a future date and two distinct known teams. Reject all extra request
  fields so post-match information cannot enter the API accidentally.
- Reconstruct state at application startup and keep prediction calls immutable.
- Package the API under `footcast.api` so editable installs, wheels, tests, and
  Uvicorn use the same import path.
- Track a canonical JSON model specification and expose its hash.

### Verification observations

- Synthetic service and endpoint tests cover probability order, normalization,
  mutation, invalid requests, holdout rejection, and response metadata.
- The real service reconstructed 34 historically observed teams with data
  cutoff `2025-05-25` and historical draw rate `0.2332`.
- A real local Arsenal-Chelsea request returned home/draw/away probabilities
  `0.5397`, `0.2332`, and `0.2272`; this is a smoke-test example, not a future
  accuracy claim.
- Uvicorn served all four endpoints successfully and response-time headers were
  present.

### Next checkpoint

Add deterministic recent-form, comparison, and head-to-head endpoints, then
build a Streamlit dashboard that calls FastAPI rather than importing model code.

## Phase 7 Checkpoint 2: Analytics and dashboard

### Concepts

- Descriptive analytics and predictive features are different. Recent form can
  help a user interpret context without silently changing what the model uses.
- Keeping the dashboard behind HTTP prevents model and data concerns from
  leaking into the presentation process.
- A team's view of a match must reverse goals and outcome when that team played
  away; the stored fixture orientation should still be preserved for
  head-to-head display.
- Product validation includes browser behavior and error states as well as
  Python functions.

### Decisions

- Load the approved train, validation, and test history once at API startup and
  share it with immutable Elo inference and read-only analytics.
- Continue to reject the 2025-26 holdout in both services.
- Limit analytics requests to 1 through 20 matches and return latest first.
- Make Streamlit call FastAPI through a small defensive standard-library HTTP
  client. Do not import Elo or read raw files in the dashboard.
- Display the data cutoff, model version, intended use, and limitations beside
  the product output.

### Verification observations

- Analytics tests cover chronology, team perspective, scores, points,
  head-to-head orientation, invalid teams, limits, and holdout rejection.
- HTTP-client tests cover URL encoding, minimal pre-match payloads, API
  validation errors, and unavailable-service errors.
- Streamlit's headless test rendered both team controls, form tables, and the
  product shell against an injected client.
- All 109 repository tests and Ruff checks passed.
- Live FastAPI and Streamlit processes loaded the real approved snapshot. An
  Arsenal-Chelsea example displayed `54.0%` home, `23.3%` draw, and `22.7%`
  away, plus current ratings, form, and head-to-head data. This verifies product
  wiring, not future accuracy.

### Outcome

Phase 7 checkpoint 2 completes the first usable local FootCast product. The
model is unchanged; this checkpoint improves accessibility, transparency, and
portfolio presentation rather than predictive performance.

## Phase 7 UI polish

### Concepts

- A portfolio dashboard needs a visual hierarchy: fixture first, forecast
  second, supporting context third, and limitations always reachable.
- Responsive validation should measure overflow and computed grid behavior,
  not rely only on shrinking a screenshot.
- Custom HTML inside Markdown must avoid retained indentation. Nested repeated
  rows can otherwise be interpreted as code blocks after the first element.
- Color can reinforce meaning, but W/D/L letters and text labels must carry the
  same information for accessibility.

### Decisions

- Keep Streamlit and the existing HTTP boundary instead of adding a second
  frontend framework for a presentation-only checkpoint.
- Create a local CSS design system with no remote fonts, JavaScript, crest
  downloads, or new runtime services.
- Use team initials as neutral visual identifiers and preserve the explicit
  educational-use language.
- Collapse matchup, probability, and Elo grids below `760px`, with the fixture
  controls provided by Streamlit's mobile sidebar drawer.
- Leave every API request, probability, rating, split, and model limitation
  unchanged.

### Verification observations

- The headless dashboard test covers the idle shell and a generated forecast.
- All 109 repository tests and Ruff checks pass.
- Live desktop testing displayed the real `54.0% / 23.3% / 22.7%`
  Arsenal-Chelsea example with the redesigned probability spectrum.
- At a `390px` viewport, all major grids collapsed to one column, the document
  width remained `390px`, and the complete hero cleared the fixed toolbar.
- Browser inspection exposed and corrected nested-row code rendering,
  percentage wrapping, and top safe-area spacing before publication.

### Outcome

FootCast now has a cohesive, responsive portfolio presentation without
changing its scientific claims. The next product milestone is public
deployment and a README demo capture; accuracy work remains a separate track.

## Phase 8 Checkpoint 1: Containers and CI smoke test

### Concepts

- Reproducible deployment includes the data snapshot, not only Python package
  versions. A serving image should prove exactly which manifest splits it uses.
- A build-time bootstrap converts a versioned manifest plus source checksums
  into an immutable runtime snapshot. Invalid or drifted data prevents the
  image from being created.
- Separate API and dashboard images preserve the HTTP product boundary and
  avoid giving the presentation process direct access to raw data.
- Health checks answer whether a process can serve traffic; a provenance smoke
  test separately verifies that it loaded the intended model-data contract.
- Container hardening can be simple and visible: use a non-root user, drop
  capabilities, and disallow privilege escalation.

### Decisions

- Add one explicit serving-data selector limited to `train`, `validation`, and
  `test`; fail on an incomplete approved history or any holdout entry.
- Build the API snapshot from the checksum-verified Phase 1 downloader and run
  the existing canonical schema validation before completing the image.
- Do not copy workstation data, reports, notebooks, virtual environments, or
  Git history into either image.
- Route Streamlit to `http://api:8000` inside Compose while retaining
  configurable host ports for developers.
- Make GitHub Actions build and start the real two-container stack after Python
  tests pass, then assert its health and `/model/info` provenance.
- Leave public hosting and runtime monitoring for the next Phase 8 checkpoint.

### Verification observations

- Ruff passes and all 116 repository tests pass.
- The local serving bootstrap validates exactly 3,800 matches through
  `2025-05-25` and reports `holdout_included=false`.
- Tests cover exact split selection, incomplete-history rejection, downloader
  isolation from holdout, non-root users, health checks, internal API routing,
  Compose hardening, and raw-data build-context exclusion.
- Docker was unavailable locally, so GitHub Actions is the authoritative build
  and live Compose verification for both images.

### Outcome

FootCast now has a portable, reviewable deployment unit and a CI gate that
tests the running product boundary. It is ready for the next Phase 8 checkpoint:
selecting a host, publishing the stack, and adding basic production monitoring.

## Phase 8 Checkpoint 2: Public deployment and monitoring

### Concepts

- Infrastructure as code makes the host configuration reviewable beside the
  application: runtime, region, plan, health endpoints, and service wiring are
  part of the repository contract.
- A generated service URL should be passed by a platform reference instead of
  copied into source. This keeps redeployments and service renames declarative.
- Process health and product correctness are different. A server can return
  HTTP 200 while loading the wrong data, so external monitoring should verify
  provenance as well as availability.
- Deployment should follow CI rather than race it. A host can wait until GitHub
  checks pass before building the new production revision.
- Account authorization is intentionally outside source control. The repo can
  describe resources, but its owner must connect GitHub to the hosting account.

### Decisions

- Select Render because its Blueprint supports both Docker services, generated
  cross-service environment values, health checks, and CI-gated auto-deploys.
- Use two public free web services in Virginia. The API remains public for the
  portfolio's OpenAPI demo; Streamlit consumes its generated external URL.
- Preserve the exact Phase 8 checkpoint 1 Dockerfiles instead of creating a
  host-specific execution path.
- Monitor `/health`, `/model/info`, and Streamlit health every six hours from
  GitHub Actions after public URL variables are configured.
- Fail monitoring if the match count, data cutoff, or holdout boundary drifts,
  even when both processes are technically reachable.

### Verification observations

- Render's current official JSON schema accepts `render.yaml`.
- Ruff passes and all 121 repository tests pass.
- Contract tests cover both service definitions, Dockerfile selection, health
  paths, free-plan region, CI-gated deploys, and generated API URL wiring.
- Monitor tests cover the healthy product, provenance drift, and network
  failure paths without calling a live deployment.
- The `footcast-production` Blueprint deployed both Docker services from merged
  commit `dc643ef`; Render reported both initial deploys live.
- The live monitor passed with 3,800 matches, cutoff `2025-05-25`, Elo reference
  version metadata, and `holdout_used=false`.
- GitHub repository variables store both public URLs, and the first manually
  triggered `Public deployment monitor` workflow passed in eight seconds.
- The workspace is Hobby, both instances are Free, paid workflow concurrency is
  zero, and the build-pipeline monthly spend limit is `$0`. Bandwidth has a
  separate included allowance and no equivalent zero-dollar cap.

### Outcome

FootCast is publicly accessible as a monitored two-service portfolio product.
Future changes merged to `main` deploy only after CI passes, so UI work can
continue without changing the model or weakening the data boundary.

## Portfolio Analytics Checkpoint

### Concepts

- A portfolio product should expose both its useful output and the evidence
  needed to judge that output. Overall accuracy alone can hide a failed class.
- Descriptive charts and predictive features are different. Recent form can
  help a visitor understand the matchup without silently changing the model.
- A frontend should receive aggregated contracts through the product API, not
  read research reports or raw CSV files directly.
- Static evaluation numbers need provenance. A test can prevent dashboard
  metrics from drifting away from the final tracked experiment.

### Decisions

- Organize the dashboard into Match Forecast, Team Analytics, and Model Insights
  instead of placing every result on one long page.
- Add one read-only portfolio endpoint for dataset coverage, outcome balance,
  current Elo leaders, and the untouched 2024-25 benchmark evidence.
- Use the existing ten-match API boundary for form and scoring charts; do not
  turn that descriptive information into new model inputs.
- Display Elo's zero draw recall and complete confusion matrix alongside its
  accuracy and log loss.
- Keep the existing model, raw-data snapshot, and prediction schema unchanged.

### Outcome

FootCast now explains team context, dataset scope, model selection, and failure
modes visually. This increases the product's portfolio value without making an
unsupported accuracy claim or weakening the API/data separation.

## Phase 9 Checkpoint 1: Assistant requirements and benchmark

### Concepts

- An LLM assistant should be evaluated against representative questions before
  prompts or providers are selected. Otherwise fluent demos can hide incorrect
  facts and routing.
- Grounding means data-dependent claims are supported by deterministic tool
  results. It does not mean the language model has memorized the project.
- Tool routing and factual correctness are separate: a model can choose the
  right tool but pass the wrong team, date, or window.
- Refusing unsupported live data is a correct product behavior, not a failure
  to be helpful.
- Cost and latency belong in the acceptance criteria before an API key is
  introduced.

### Decisions

- Define five initial single-tool question families and postpone bounded
  multi-tool answers until routing is reliable.
- Create 42 balanced cases with exact expected arguments and dynamic evidence
  requirements rather than copying mutable numeric answers into the benchmark.
- Require explicit separation of observed history, model predictions, general
  explanations, and refusals.
- Include contextual follow-ups, entity normalization, the deployed model's
  draw weakness, betting pressure, prompt injection, and secret exfiltration.
- Set initial routing, groundedness, safety, latency, tool-call, and cost gates
  without choosing an LLM provider.

### Verification observations

- Forty-five focused tests validate the benchmark's JSONL syntax, 42 unique
  cases, balanced categories, exact tool argument shapes, evidence metadata,
  contextual coverage, risk coverage, and future prediction dates.
- No provider package, API key, chat endpoint, or assistant runtime was added.

### Outcome

FootCast now has a measurable contract for what its future assistant may say
and how success will be judged. The next checkpoint can implement and test the
five typed read-only tools without involving an LLM.

## Phase 9 Checkpoint 2: Typed read-only tools

### Concepts

- Tool calling separates language interpretation from deterministic
  calculation. An LLM may choose a tool later, but it cannot redefine its
  validation or compute its evidence.
- Strict schemas turn a request into a small auditable input contract and
  reject accidental or malicious extra fields.
- Provenance is part of the result, not optional prose added by the LLM.
- A shared data cutoff prevents a response from combining two valid but
  inconsistent snapshots.
- Provider-neutral tools keep model selection reversible and the core product
  testable without network access or usage charges.

### Decisions

- Implement five single-purpose tools over the existing immutable Elo and
  analytics services rather than duplicate their calculations.
- Publish strict Pydantic JSON Schemas and a read-only flag through one catalog.
- Use one dispatcher that validates exact arguments and converts domain errors
  into a safe tool error.
- Return typed evidence envelopes with sources and timezone-aware timestamps.
- Limit model explanations and metric definitions to the benchmarked approved
  topics; do not let the deterministic layer improvise answers.
- Keep the provider client, prompt policy, rate limits, conversation state, and
  chat UI out of this checkpoint.

### Verification observations

- Focused tests cover tool schemas, routing, validation, provenance,
  serialization, shared-cutoff enforcement, prediction immutability, every
  explanation and metric topic, and compatibility with all tool-backed
  benchmark cases.
- Calibration explanation values remain synchronized with the tracked Phase 5
  report.
- The tools require no provider dependency, key, network request, or new
  deployment environment value.

### Outcome

FootCast now has five deterministic assistant capabilities that can be tested
independently of language generation. The next checkpoint can connect a
configurable server-side LLM client and policy without giving it direct access
to model code, raw data, or credentials.
