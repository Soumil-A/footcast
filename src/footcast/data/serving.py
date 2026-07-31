"""Acquire and validate only data approved for the serving image."""

from __future__ import annotations

import argparse
from pathlib import Path

from footcast.config import DATA_SPLIT, SERVING_SPLITS
from footcast.data.download import DEFAULT_RAW_DIR, download_one
from footcast.data.manifest import DownloadSpec, load_manifest
from footcast.data.matches import load_match_statistics


def approved_serving_specs(
    specs: tuple[DownloadSpec, ...] | None = None,
) -> tuple[DownloadSpec, ...]:
    """Return the exact non-holdout history allowed in a serving image."""
    manifest = specs or load_manifest()
    selected = tuple(spec for spec in manifest if spec.split in SERVING_SPLITS)
    expected_seasons = DATA_SPLIT.train + DATA_SPLIT.validation + DATA_SPLIT.test
    actual_seasons = tuple(spec.season for spec in selected)
    if actual_seasons != expected_seasons:
        raise ValueError(
            "Serving data must exactly cover train, validation, and test seasons"
        )
    if any(spec.split == "holdout" for spec in selected):
        raise ValueError("Holdout data cannot enter the serving image")
    return selected


def prepare_serving_data(
    raw_dir: Path = DEFAULT_RAW_DIR,
    *,
    specs: tuple[DownloadSpec, ...] | None = None,
) -> dict[str, object]:
    """Download, checksum, and validate the approved serving snapshot."""
    selected = approved_serving_specs(specs)
    for spec in selected:
        download_one(spec, raw_dir)
    matches = load_match_statistics(
        raw_dir,
        specs=selected,
        splits=SERVING_SPLITS,
    )
    return {
        "status": "passed",
        "seasons": [spec.season for spec in selected],
        "splits": sorted({spec.split for spec in selected}),
        "completed_matches": int(len(matches)),
        "data_cutoff": str(matches["match_date"].max()),
        "holdout_included": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    args = parser.parse_args()
    report = prepare_serving_data(args.raw_dir)
    print(
        f"Prepared {report['completed_matches']} approved matches through "
        f"{report['data_cutoff']}; holdout_included=false."
    )


if __name__ == "__main__":
    main()
