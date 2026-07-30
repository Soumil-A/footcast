"""Load validated match statistics for explicitly approved dataset splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from footcast.data.download import DEFAULT_RAW_DIR
from footcast.data.manifest import DownloadSpec, load_manifest
from footcast.data.validate import validate_season

MATCH_STAT_COLUMN_MAP = {
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellow_cards",
    "AY": "away_yellow_cards",
    "HR": "home_red_cards",
    "AR": "away_red_cards",
}


def prepare_match_statistics(
    frame: pd.DataFrame, spec: DownloadSpec
) -> pd.DataFrame:
    """Validate one raw season and attach stable descriptive statistics."""
    validated = validate_season(frame, spec)
    source = pd.DataFrame(
        {
            "season": spec.season,
            "match_date": pd.to_datetime(
                frame["Date"], dayfirst=True, format="mixed"
            ).dt.strftime("%Y-%m-%d"),
            "home_team": frame["HomeTeam"].astype(str).str.strip(),
            "away_team": frame["AwayTeam"].astype(str).str.strip(),
        }
    )
    for source_column, canonical_column in MATCH_STAT_COLUMN_MAP.items():
        source[canonical_column] = (
            pd.to_numeric(frame[source_column], errors="coerce")
            if source_column in frame
            else pd.NA
        )

    matches = validated.matches.merge(
        source,
        on=["season", "match_date", "home_team", "away_team"],
        how="left",
        validate="one_to_one",
    )
    matches["split"] = spec.split
    return matches


def load_match_statistics(
    raw_dir: Path = DEFAULT_RAW_DIR,
    *,
    specs: tuple[DownloadSpec, ...] | None = None,
    splits: frozenset[str],
) -> pd.DataFrame:
    """Load only requested splits, validating each raw season before use."""
    selected = tuple(
        spec
        for spec in (specs or load_manifest())
        if spec.split in splits
    )
    if not selected:
        raise ValueError("No manifest seasons match the requested splits")

    matches: list[pd.DataFrame] = []
    for spec in selected:
        path = raw_dir / spec.filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `python -m footcast.data.pipeline` first."
            )
        matches.append(prepare_match_statistics(pd.read_csv(path), spec))
    return pd.concat(matches, ignore_index=True).sort_values(
        ["match_date", "home_team", "away_team"], ignore_index=True
    )
