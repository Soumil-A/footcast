import pandas as pd

from footcast.data.pipeline import schema_drift
from footcast.data.validate import ValidatedSeason


def _validated(columns: tuple[str, ...]) -> ValidatedSeason:
    return ValidatedSeason(
        matches=pd.DataFrame(),
        source_columns=columns,
        teams=(),
        missing_values={},
    )


def test_schema_drift_records_added_and_removed_source_columns() -> None:
    previous = _validated(("Date", "HomeTeam", "LegacyOdds"))
    current = _validated(("Date", "HomeTeam", "Time", "NewOdds"))

    added, removed = schema_drift(current, previous)

    assert added == ["NewOdds", "Time"]
    assert removed == ["LegacyOdds"]
