# Pre-Match Feature Contract

Phase 3 converts completed match history into values available immediately
before each target kickoff. The generated development table covers 2015-16
through 2023-24. Test and holdout seasons remain excluded.

Every team feature appears with both `home_` and `away_` prefixes.

| Feature | Definition | Source columns | Available | Leakage risk | Missing-history behavior |
| --- | --- | --- | --- | --- | --- |
| `history_matches` | All earlier observed matches for the team | Date, teams | Before kickoff | Current match counted | `0`; row retained |
| `rolling_matches` | Number of earlier matches used, capped at five | Date, teams | Before kickoff | Current match counted | `0`; row retained |
| `form_points_last_5` | Points from up to five completed prior matches | FTR | After prior full time | Current result included | `0` plus window count |
| `form_wins_last_5` | Wins from up to five completed prior matches | FTR | After prior full time | Current result included | `0` plus window count |
| `goals_for_last_5` | Goals scored in up to five completed prior matches | FTHG, FTAG | After prior full time | Current goals included | `0` plus window count |
| `goals_against_last_5` | Goals conceded in up to five completed prior matches | FTHG, FTAG | After prior full time | Current goals included | `0` plus window count |
| `shots_last_5` | Shots in up to five completed prior matches | HS, AS | After prior full time | Current shots included | `0` plus window count |
| `shots_on_target_last_5` | Target shots in up to five completed prior matches | HST, AST | After prior full time | Current target shots included | `0` plus window count |
| `venue_matches_last_5` | Prior same-role home/away matches used, capped at five | Teams | Before kickoff | Wrong role assigned | `0` |
| `venue_points_last_5` | Points from up to five earlier matches at the same role | Teams, FTR | After prior full time | Current result included | `0` plus venue count |
| `days_since_previous_match` | Days since the team's previous observed match | Date | On match date | Incorrect ordering | Missing on first appearance |
| `season_matches_played` | Earlier completed matches in the current season | Season, teams | Before kickoff | Final totals used | Resets to `0` |
| `season_points` | Points earned earlier in the current season | Season, FTR | After prior full time | Final table used | Resets to `0` |
| `expanding_goals_for_mean` | Mean goals scored across all earlier observed matches | FTHG, FTAG | After prior full time | Current/future goals included | `0` plus history count |
| `expanding_goals_against_mean` | Mean goals conceded across all earlier observed matches | FTHG, FTAG | After prior full time | Current/future goals included | `0` plus history count |
| `elo` | Rating snapshot before kickoff | Teams, FTR | Before kickoff | Current result update applied early | Starts at `1500` |

## Matchup differences

Each difference is calculated as the home pre-match value minus the away
pre-match value:

- `elo_difference`
- `form_points_difference`
- `goals_scored_difference`
- `goals_conceded_difference`
- `rest_days_difference`
- `shots_on_target_difference`

`rest_days_difference` remains missing if either team has no previous observed
match. The row is never discarded.

## Window and season behavior

- General five-match form, expanding history, rest, and Elo carry across season
  boundaries because those earlier completed matches were already known.
- `season_matches_played` and `season_points` reset at every season boundary.
- Returning teams retain their previous available Premier League history and
  Elo. Completely unseen teams start with empty history and Elo `1500`.
- Five-match quantities are sums. The paired window counts reveal whether they
  use zero, one, or all five prior matches.

## Elo update order

The rating system uses an initial rating of `1500`, `K=20`, a `400`-point
rating scale, and a fixed 65-point home adjustment when calculating expected
score.

For each match:

1. Record both teams' ratings as pre-match features.
2. Calculate the expected home score.
3. Observe the completed result.
4. Update both ratings by equal and opposite amounts.

The result from a match can affect only later matches.
