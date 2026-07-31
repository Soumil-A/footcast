"""Tests for the deployment-safe approved data bootstrap."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from footcast.config import DATA_SPLIT, SERVING_SPLITS
from footcast.data.manifest import load_manifest
from footcast.data.serving import approved_serving_specs, prepare_serving_data


def test_approved_serving_specs_exclude_holdout() -> None:
    selected = approved_serving_specs(load_manifest())

    assert {spec.split for spec in selected} == SERVING_SPLITS
    assert tuple(spec.season for spec in selected) == (
        DATA_SPLIT.train + DATA_SPLIT.validation + DATA_SPLIT.test
    )
    assert DATA_SPLIT.holdout[0] not in {spec.season for spec in selected}


def test_approved_serving_specs_reject_incomplete_history() -> None:
    incomplete = load_manifest()[:-2]

    with pytest.raises(ValueError, match="exactly cover"):
        approved_serving_specs(incomplete)


def test_prepare_serving_data_downloads_and_validates_only_approved_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[str] = []

    def fake_download(spec, raw_dir):
        requested.append(spec.season)
        return raw_dir / spec.filename

    matches = pd.DataFrame(
        {
            "match_date": ["2025-05-25"],
            "home_team": ["Alpha"],
            "away_team": ["Beta"],
        }
    )

    def fake_load(raw_dir, *, specs, splits):
        assert raw_dir == tmp_path
        assert {spec.split for spec in specs} == SERVING_SPLITS
        assert splits == SERVING_SPLITS
        return matches

    monkeypatch.setattr("footcast.data.serving.download_one", fake_download)
    monkeypatch.setattr(
        "footcast.data.serving.load_match_statistics", fake_load
    )

    report = prepare_serving_data(tmp_path, specs=load_manifest())

    assert requested == list(
        DATA_SPLIT.train + DATA_SPLIT.validation + DATA_SPLIT.test
    )
    assert DATA_SPLIT.holdout[0] not in requested
    assert report == {
        "status": "passed",
        "seasons": requested,
        "splits": ["test", "train", "validation"],
        "completed_matches": 1,
        "data_cutoff": "2025-05-25",
        "holdout_included": False,
    }
