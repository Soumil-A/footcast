"""End-to-end Phase 1 pipeline: download, validate, canonicalize, and audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from footcast.data.download import DEFAULT_RAW_DIR, download_all
from footcast.data.manifest import DEFAULT_MANIFEST, DownloadSpec, load_manifest
from footcast.data.validate import CANONICAL_COLUMNS, ValidatedSeason, validate_season

PROJECT_ROOT = DEFAULT_MANIFEST.parents[1]
DEFAULT_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "matches.csv"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports" / "data_quality.json"
DEFAULT_REPORT_MARKDOWN = PROJECT_ROOT / "reports" / "data_quality.md"


def schema_drift(
    current: ValidatedSeason, previous: ValidatedSeason | None
) -> tuple[list[str], list[str]]:
    """Return source columns added and removed relative to a prior season."""
    if previous is None:
        return [], []
    current_columns = set(current.source_columns)
    previous_columns = set(previous.source_columns)
    return (
        sorted(current_columns - previous_columns),
        sorted(previous_columns - current_columns),
    )


def _season_report(
    spec: DownloadSpec,
    validated: ValidatedSeason,
    previous: ValidatedSeason | None,
) -> dict[str, object]:
    matches = validated.matches
    added_columns, removed_columns = schema_drift(validated, previous)
    current_teams = set(validated.teams)
    previous_teams = set(previous.teams) if previous else set()
    return {
        "season": spec.season,
        "split": spec.split,
        "source": {
            "url": spec.url,
            "filename": spec.filename,
            "sha256": spec.sha256,
        },
        "rows": len(matches),
        "source_columns": len(validated.source_columns),
        "date_min": matches["match_date"].min(),
        "date_max": matches["match_date"].max(),
        "teams": list(validated.teams),
        "teams_added_since_previous_season": sorted(current_teams - previous_teams)
        if previous
        else [],
        "teams_removed_since_previous_season": sorted(previous_teams - current_teams)
        if previous
        else [],
        "missing_values": validated.missing_values,
        "schema_added_since_previous_season": added_columns,
        "schema_removed_since_previous_season": removed_columns,
        "validation": "passed",
    }


def build_quality_report(
    validated: list[ValidatedSeason],
) -> dict[str, object]:
    """Build a JSON-serializable quality report across seasons."""
    specs = load_manifest()
    seasons: list[dict[str, object]] = []
    for index, (spec, season) in enumerate(zip(specs, validated, strict=True)):
        previous = validated[index - 1] if index else None
        seasons.append(_season_report(spec, season, previous))
    all_matches = pd.concat(
        [season.matches for season in validated], ignore_index=True
    )
    return {
        "status": "passed",
        "source": "Football-Data Premier League (E0)",
        "canonical_columns": list(CANONICAL_COLUMNS),
        "total_rows": len(all_matches),
        "season_count": len(seasons),
        "seasons": seasons,
    }


def render_markdown(report: dict[str, object]) -> str:
    """Render the essential audit findings as a reviewable Markdown report."""
    lines = [
        "# FootCast Data-Quality Report",
        "",
        f"**Status:** {str(report['status']).upper()}",
        "",
        (
            f"Validated {report['total_rows']} Premier League matches across "
            f"{report['season_count']} seasons."
        ),
        "",
        "## Canonical schema",
        "",
        "| Column | Meaning |",
        "| --- | --- |",
        "| `season` | Declared season identifier (`YYYY-YY`) |",
        "| `match_date` | Match date in ISO `YYYY-MM-DD` format |",
        "| `home_team` | Source home-team name, whitespace checked |",
        "| `away_team` | Source away-team name, whitespace checked |",
        "| `full_time_home_goals` | Nonnegative integer home goals |",
        "| `full_time_away_goals` | Nonnegative integer away goals |",
        "| `result` | `home_win`, `draw`, or `away_win` |",
        "",
        "## Season audit",
        "",
        "| Season | Split | Rows | Source columns | Date range | Missing cells |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    seasons = report["seasons"]
    assert isinstance(seasons, list)
    for season in seasons:
        missing = season["missing_values"]
        assert isinstance(missing, dict)
        lines.append(
            f"| {season['season']} | {season['split']} | {season['rows']} | "
            f"{season['source_columns']} | {season['date_min']} to "
            f"{season['date_max']} | {sum(missing.values())} |"
        )

    lines.extend(
        [
            "",
            "All seasons passed required-column, date, team, score, result, "
            "duplicate-fixture, row-count, team-count, and season-boundary checks.",
            "Missing-cell totals include optional source fields; required canonical "
            "fields contain no missing values.",
            "",
            "## Schema drift",
            "",
            "Optional Football-Data fields change over time. The canonical seven "
            "columns remain stable; additions/removals below are source-only fields.",
            "",
        ]
    )
    for season in seasons:
        added = season["schema_added_since_previous_season"]
        removed = season["schema_removed_since_previous_season"]
        if added or removed:
            lines.append(f"### {season['season']}")
            lines.append("")
            lines.append(f"- Added: {', '.join(added) if added else 'none'}")
            lines.append(f"- Removed: {', '.join(removed) if removed else 'none'}")
            lines.append("")

    lines.extend(
        [
            "## Team movement",
            "",
            "Added and removed names are audit candidates for promotion/relegation; "
            "they are set differences, not independently verified "
            "league-status claims.",
            "",
            "| Season | Added vs previous | Removed vs previous |",
            "| --- | --- | --- |",
        ]
    )
    for season in seasons[1:]:
        added = ", ".join(season["teams_added_since_previous_season"])
        removed = ", ".join(season["teams_removed_since_previous_season"])
        lines.append(f"| {season['season']} | {added} | {removed} |")

    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "python -m footcast.data.pipeline",
            "```",
            "",
            "The downloader verifies every file against the versioned manifest and "
            "refuses to overwrite a raw file whose bytes differ.",
            "",
        ]
    )
    return "\n".join(lines)


def run_pipeline(
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_path: Path = DEFAULT_PROCESSED_PATH,
    report_json: Path = DEFAULT_REPORT_JSON,
    report_markdown: Path = DEFAULT_REPORT_MARKDOWN,
) -> dict[str, object]:
    """Acquire all source files and write reproducible Phase 1 artifacts."""
    paths = download_all(raw_dir)
    validated = [
        validate_season(pd.read_csv(path), spec)
        for path, spec in zip(paths, load_manifest(), strict=True)
    ]
    matches = pd.concat(
        [season.matches for season in validated], ignore_index=True
    )
    matches = matches.sort_values(
        ["match_date", "home_team", "away_team"], ignore_index=True
    )
    report = build_quality_report(validated)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_markdown.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(processed_path, index=False)
    report_json.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    report_markdown.write_text(render_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report = run_pipeline()
    print(
        f"Validated {report['total_rows']} matches across "
        f"{report['season_count']} seasons."
    )


if __name__ == "__main__":
    main()
