# FootCast Data-Quality Report

**Status:** PASSED

Validated 4180 Premier League matches across 11 seasons.

## Canonical schema

| Column | Meaning |
| --- | --- |
| `season` | Declared season identifier (`YYYY-YY`) |
| `match_date` | Match date in ISO `YYYY-MM-DD` format |
| `home_team` | Source home-team name, whitespace checked |
| `away_team` | Source away-team name, whitespace checked |
| `full_time_home_goals` | Nonnegative integer home goals |
| `full_time_away_goals` | Nonnegative integer away goals |
| `result` | `home_win`, `draw`, or `away_win` |

## Season audit

| Season | Split | Rows | Source columns | Date range | Missing cells |
| --- | --- | ---: | ---: | --- | ---: |
| 2015-16 | train | 380 | 65 | 2015-08-08 to 2016-05-17 | 3 |
| 2016-17 | train | 380 | 65 | 2016-08-13 to 2017-05-21 | 0 |
| 2017-18 | train | 380 | 65 | 2017-08-11 to 2018-05-13 | 0 |
| 2018-19 | train | 380 | 62 | 2018-08-10 to 2019-05-12 | 0 |
| 2019-20 | train | 380 | 106 | 2019-08-09 to 2020-07-26 | 0 |
| 2020-21 | train | 380 | 106 | 2020-09-12 to 2021-05-23 | 0 |
| 2021-22 | train | 380 | 106 | 2021-08-13 to 2022-05-22 | 13 |
| 2022-23 | train | 380 | 106 | 2022-08-05 to 2023-05-28 | 4 |
| 2023-24 | validation | 380 | 106 | 2023-08-11 to 2024-05-19 | 1166 |
| 2024-25 | test | 380 | 120 | 2024-08-16 to 2025-05-25 | 1440 |
| 2025-26 | holdout | 380 | 132 | 2025-08-15 to 2026-05-24 | 3971 |

All seasons passed required-column, date, team, score, result, duplicate-fixture, row-count, team-count, and season-boundary checks.
Missing-cell totals include optional source fields; required canonical fields contain no missing values.

## Schema drift

Optional Football-Data fields change over time. The canonical seven columns remain stable; additions/removals below are source-only fields.

### 2018-19

- Added: none
- Removed: LBA, LBD, LBH

### 2019-20

- Added: AHCh, AHh, Avg<2.5, Avg>2.5, AvgA, AvgAHA, AvgAHH, AvgC<2.5, AvgC>2.5, AvgCA, AvgCAHA, AvgCAHH, AvgCD, AvgCH, AvgD, AvgH, B365<2.5, B365>2.5, B365AHA, B365AHH, B365C<2.5, B365C>2.5, B365CA, B365CAHA, B365CAHH, B365CD, B365CH, BWCA, BWCD, BWCH, IWCA, IWCD, IWCH, Max<2.5, Max>2.5, MaxA, MaxAHA, MaxAHH, MaxC<2.5, MaxC>2.5, MaxCA, MaxCAHA, MaxCAHH, MaxCD, MaxCH, MaxD, MaxH, P<2.5, P>2.5, PAHA, PAHH, PC<2.5, PC>2.5, PCAHA, PCAHH, Time, VCCA, VCCD, VCCH, WHCA, WHCD, WHCH
- Removed: Bb1X2, BbAH, BbAHh, BbAv<2.5, BbAv>2.5, BbAvA, BbAvAHA, BbAvAHH, BbAvD, BbAvH, BbMx<2.5, BbMx>2.5, BbMxA, BbMxAHA, BbMxAHH, BbMxD, BbMxH, BbOU

### 2024-25

- Added: 1XBA, 1XBCA, 1XBCD, 1XBCH, 1XBD, 1XBH, BFA, BFCA, BFCD, BFCH, BFD, BFE<2.5, BFE>2.5, BFEA, BFEAHA, BFEAHH, BFEC<2.5, BFEC>2.5, BFECA, BFECAHA, BFECAHH, BFECD, BFECH, BFED, BFEH, BFH
- Removed: IWA, IWCA, IWCD, IWCH, IWD, IWH, VCA, VCCA, VCCD, VCCH, VCD, VCH

### 2025-26

- Added: BFDA, BFDCA, BFDCD, BFDCH, BFDD, BFDH, BMGMA, BMGMCA, BMGMCD, BMGMCH, BMGMD, BMGMH, BVA, BVCA, BVCD, BVCH, BVD, BVH, CLA, CLCA, CLCD, CLCH, CLD, CLH, LBA, LBCA, LBCD, LBCH, LBD, LBH
- Removed: 1XBA, 1XBCA, 1XBCD, 1XBCH, 1XBD, 1XBH, BFA, BFCA, BFCD, BFCH, BFD, BFH, WHA, WHCA, WHCD, WHCH, WHD, WHH

## Team movement

Added and removed names are audit candidates for promotion/relegation; they are set differences, not independently verified league-status claims.

| Season | Added vs previous | Removed vs previous |
| --- | --- | --- |
| 2016-17 | Burnley, Hull, Middlesbrough | Aston Villa, Newcastle, Norwich |
| 2017-18 | Brighton, Huddersfield, Newcastle | Hull, Middlesbrough, Sunderland |
| 2018-19 | Cardiff, Fulham, Wolves | Stoke, Swansea, West Brom |
| 2019-20 | Aston Villa, Norwich, Sheffield United | Cardiff, Fulham, Huddersfield |
| 2020-21 | Fulham, Leeds, West Brom | Bournemouth, Norwich, Watford |
| 2021-22 | Brentford, Norwich, Watford | Fulham, Sheffield United, West Brom |
| 2022-23 | Bournemouth, Fulham, Nott'm Forest | Burnley, Norwich, Watford |
| 2023-24 | Burnley, Luton, Sheffield United | Leeds, Leicester, Southampton |
| 2024-25 | Ipswich, Leicester, Southampton | Burnley, Luton, Sheffield United |
| 2025-26 | Burnley, Leeds, Sunderland | Ipswich, Leicester, Southampton |

## Reproduction

```bash
python -m footcast.data.pipeline
```

The downloader verifies every file against the versioned manifest and refuses to overwrite a raw file whose bytes differ.
