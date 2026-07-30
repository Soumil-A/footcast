from footcast import DATA_SPLIT, PROJECT_NAME


def test_project_name() -> None:
    assert PROJECT_NAME == "FootCast"


def test_season_split_is_chronological_and_disjoint() -> None:
    seasons = DATA_SPLIT.all_seasons()

    assert seasons == tuple(sorted(seasons))
    assert len(seasons) == len(set(seasons))
    assert DATA_SPLIT.validation == ("2023-24",)
    assert DATA_SPLIT.test == ("2024-25",)
    assert DATA_SPLIT.holdout == ("2025-26",)
