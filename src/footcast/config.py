"""Project-wide constants and the chronological evaluation contract."""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_NAME = "FootCast"
SOURCE_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(
    os.getenv("FOOTCAST_PROJECT_ROOT", SOURCE_PROJECT_ROOT)
).resolve()


@dataclass(frozen=True)
class SeasonSplit:
    """Seasons reserved for each stage of model development."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    holdout: tuple[str, ...]

    def all_seasons(self) -> tuple[str, ...]:
        """Return all seasons in chronological order."""
        return self.train + self.validation + self.test + self.holdout


DATA_SPLIT = SeasonSplit(
    train=(
        "2015-16",
        "2016-17",
        "2017-18",
        "2018-19",
        "2019-20",
        "2020-21",
        "2021-22",
        "2022-23",
    ),
    validation=("2023-24",),
    test=("2024-25",),
    holdout=("2025-26",),
)

SERVING_SPLITS = frozenset({"train", "validation", "test"})
