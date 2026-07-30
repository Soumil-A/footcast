from datetime import date
from io import BytesIO
from pathlib import Path

import pytest

from footcast.data.download import DownloadError, download_one
from footcast.data.manifest import DownloadSpec


def _spec(sha256: str) -> DownloadSpec:
    return DownloadSpec(
        season="2024-25",
        split="test",
        url="https://example.invalid/source.csv",
        filename="season.csv",
        sha256=sha256,
        date_min=date(2024, 7, 1),
        date_max=date(2025, 7, 31),
        expected_rows=1,
        expected_teams=2,
    )


def test_existing_checksum_identical_raw_file_is_reused(tmp_path: Path) -> None:
    raw_file = tmp_path / "season.csv"
    raw_file.write_bytes(b"immutable bytes")
    digest = "59d8792018a51a408d2738f31eedebd6fe9926cc4260fa168a38710bc51d7e30"

    assert download_one(_spec(digest), tmp_path) == raw_file
    assert raw_file.read_bytes() == b"immutable bytes"


def test_existing_different_raw_file_is_never_overwritten(tmp_path: Path) -> None:
    raw_file = tmp_path / "season.csv"
    raw_file.write_bytes(b"user-owned raw bytes")

    with pytest.raises(DownloadError, match="Refusing to overwrite raw data"):
        download_one(_spec("0" * 64), tmp_path)

    assert raw_file.read_bytes() == b"user-owned raw bytes"


def test_new_download_with_wrong_checksum_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = BytesIO(b"unexpected upstream bytes")
    monkeypatch.setattr(
        "footcast.data.download.urllib.request.urlopen",
        lambda *args, **kwargs: source,
    )

    with pytest.raises(DownloadError, match="Checksum mismatch"):
        download_one(_spec("0" * 64), tmp_path)

    assert not (tmp_path / "season.csv").exists()
    assert not (tmp_path / "season.csv.part").exists()
