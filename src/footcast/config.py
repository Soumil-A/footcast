"""Project-wide constants and the chronological evaluation contract."""

from dataclasses import dataclass

PROJECT_NAME = "FootCast"


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
