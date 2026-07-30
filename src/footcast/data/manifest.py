"""Typed access to the versioned Football-Data download manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from footcast.config import DATA_SPLIT

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "download_manifest.json"


@dataclass(frozen=True)
class DownloadSpec:
    """One immutable source file and its validation expectations."""

    season: str
    split: str
    url: str
    filename: str
    sha256: str
    date_min: date
    date_max: date
    expected_rows: int
    expected_teams: int


def load_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[DownloadSpec, ...]:
    """Load and validate all season download specifications."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = tuple(
        DownloadSpec(
            season=item["season"],
            split=item["split"],
            url=item["url"],
            filename=item["filename"],
            sha256=item["sha256"],
            date_min=date.fromisoformat(item["date_min"]),
            date_max=date.fromisoformat(item["date_max"]),
            expected_rows=item["expected_rows"],
            expected_teams=item["expected_teams"],
        )
        for item in payload["seasons"]
    )

    seasons = tuple(item.season for item in entries)
    if seasons != DATA_SPLIT.all_seasons():
        raise ValueError(
            "Manifest seasons must exactly match DATA_SPLIT in chronological order"
        )
    if len({item.filename for item in entries}) != len(entries):
        raise ValueError("Manifest filenames must be unique")
    if len({item.url for item in entries}) != len(entries):
        raise ValueError("Manifest URLs must be unique")
    if any(len(item.sha256) != 64 for item in entries):
        raise ValueError("Every manifest entry must contain a SHA-256 digest")
    return entries
