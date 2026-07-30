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
