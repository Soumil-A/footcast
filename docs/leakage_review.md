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
| Pre-match Elo difference | Before kickoff | Elo updated too early | Planned |
| Points in previous five matches | Before kickoff | Current result included | Planned |
| Goals scored in previous five matches | Before kickoff | Current goals included | Planned |
| Goals conceded in previous five matches | Before kickoff | Current goals included | Planned |
| Days since previous match | Before kickoff | Incorrect match ordering | Planned |
| Season-to-date points | Before kickoff | Final table used | Planned |

No feature is approved until its calculation and focused leakage test exist.

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

Phase 3 may propose shifted historical versions of match statistics. Each one
must be calculated strictly before the target kickoff and receive a
hand-calculated leakage test.
