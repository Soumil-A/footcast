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
