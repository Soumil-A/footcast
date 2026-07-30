from footcast.config import DATA_SPLIT
from footcast.data.manifest import load_manifest


def test_manifest_covers_configured_seasons_in_order() -> None:
    manifest = load_manifest()

    assert tuple(item.season for item in manifest) == DATA_SPLIT.all_seasons()
    assert {item.split for item in manifest} == {
        "train",
        "validation",
        "test",
        "holdout",
    }
    assert all(
        item.url.startswith("https://www.football-data.co.uk/")
        for item in manifest
    )
    assert all(item.expected_rows == 380 for item in manifest)
    assert all(item.expected_teams == 20 for item in manifest)
