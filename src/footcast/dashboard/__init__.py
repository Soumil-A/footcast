"""Streamlit presentation layer for the FootCast API."""

from footcast.dashboard.client import FootCastApiClient, FootCastApiError

__all__ = ["FootCastApiClient", "FootCastApiError"]
