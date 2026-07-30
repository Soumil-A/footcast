"""Baseline and learned match-outcome models."""

from footcast.models.baselines import (
    AlwaysHomeBaseline,
    EloBaseline,
    LogisticRegressionBaseline,
    MajorityClassBaseline,
)

__all__ = [
    "AlwaysHomeBaseline",
    "EloBaseline",
    "LogisticRegressionBaseline",
    "MajorityClassBaseline",
]
