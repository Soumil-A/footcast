"""Download raw Football-Data files without silently changing existing bytes."""

from __future__ import annotations

import hashlib
import shutil
import ssl
import urllib.request
from pathlib import Path

import certifi

from footcast.data.manifest import DEFAULT_MANIFEST, DownloadSpec, load_manifest

DEFAULT_RAW_DIR = DEFAULT_MANIFEST.parent / "raw"
TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())


class DownloadError(RuntimeError):
    """Raised when an immutable raw-data guarantee cannot be met."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(spec: DownloadSpec, raw_dir: Path = DEFAULT_RAW_DIR) -> Path:
    """Download one manifest entry atomically and verify its exact bytes."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / spec.filename

    if destination.exists():
        actual = sha256_file(destination)
        if actual != spec.sha256:
            raise DownloadError(
                f"{destination} already exists with SHA-256 {actual}; "
                f"expected {spec.sha256}. Refusing to overwrite raw data."
            )
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(
            spec.url, timeout=60, context=TLS_CONTEXT
        ) as response:
            with temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        actual = sha256_file(temporary)
        if actual != spec.sha256:
            raise DownloadError(
                f"Checksum mismatch for {spec.season}: expected {spec.sha256}, "
                f"received {actual}. The upstream file may have changed."
            )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def download_all(raw_dir: Path = DEFAULT_RAW_DIR) -> tuple[Path, ...]:
    """Download every manifest file, reusing only checksum-identical files."""
    return tuple(download_one(spec, raw_dir) for spec in load_manifest())
