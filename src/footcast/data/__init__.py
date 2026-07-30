"""Reproducible acquisition and validation for Football-Data match files."""

from footcast.data.manifest import DownloadSpec, load_manifest
from footcast.data.validate import DataValidationError, validate_season

__all__ = [
    "DataValidationError",
    "DownloadSpec",
    "load_manifest",
    "validate_season",
]
