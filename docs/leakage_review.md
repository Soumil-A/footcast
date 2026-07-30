# Leakage Review

This document is the approval checklist for every model feature.

## Required questions

Before adding a feature, record:

1. What exactly does the feature represent?
2. Which raw fields produce it?
3. At what time does the information become available?
4. Could it contain information from the match being predicted?
5. Could it contain information from a later match or final season result?
6. How is missing history handled?
7. Which automated test proves that the current match is excluded?

## Initial feature register

| Feature | Availability | Leakage risk | Status |
| --- | --- | --- | --- |
| Pre-match Elo difference | Before kickoff | Elo updated too early | Approved |
| Points in previous five matches | Before kickoff | Current result included | Approved |
| Goals scored in previous five matches | Before kickoff | Current goals included | Approved |
| Goals conceded in previous five matches | Before kickoff | Current goals included | Approved |
| Days since previous match | Before kickoff | Incorrect match ordering | Approved |
| Season-to-date points | Before kickoff | Final table used | Approved |

Approval means the calculation exists and its focused leakage tests pass. See
`docs/features.md` for the complete contract and missing-history behavior.

## Phase 2 exploratory-use register

| Quantity | Descriptive use | Direct pre-match use | Reason |
| --- | --- | --- | --- |
| Current match result/goals | Allowed | Forbidden | Known only after the match |
| Current shots/target shots | Allowed | Forbidden | Accumulate after kickoff |
| Current corners/fouls/cards | Allowed | Forbidden | Accumulate after kickoff |
| Final season points per match | Allowed | Forbidden | Uses matches that were future at earlier dates |
| Full-window head-to-head mean | Allowed | Forbidden | Includes later meetings |
| Source-column missingness | Allowed | Not a match feature | Dataset metadata |
| Promotion-candidate label | Allowed | Not yet approved | Inferred from adjacent team sets |

Phase 3 implements shifted historical versions of match statistics. Each is
calculated strictly before the target kickoff and has a hand-calculated leakage
test.

## Phase 3 update-order guarantee

For every chronologically ordered match, the pipeline:

1. Reads the two teams' existing completed-match histories.
2. Records all pre-match feature values.
3. Records the target result separately.
4. Adds the completed match to both histories.
5. Updates Elo from the completed result.

Tests prove that the first match has empty history, the second can use only the
first, the sixth uses the preceding five rather than itself, home and away
statistics are assigned correctly, season counters reset, and Elo changes only
for later matches.
