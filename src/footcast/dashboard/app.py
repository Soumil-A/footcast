"""Portfolio-facing Streamlit dashboard backed only by the FootCast API."""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from footcast.dashboard.client import FootCastApiClient, FootCastApiError

API_URL = os.getenv("FOOTCAST_API_URL", "http://127.0.0.1:8000")
RESULT_LABELS = {
    "home_win": "Home win",
    "draw": "Draw",
    "away_win": "Away win",
}


@st.cache_data(ttl=60, show_spinner=False)
def load_reference_data(api_url: str) -> tuple[dict[str, Any], dict[str, Any]]:
    client = FootCastApiClient(api_url)
    return client.teams(), client.model_info()


@st.cache_data(ttl=60, show_spinner=False)
def load_analytics(
    api_url: str, home_team: str, away_team: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    client = FootCastApiClient(api_url)
    return (
        client.compare(home_team, away_team, limit=5),
        client.head_to_head(home_team, away_team, limit=10),
    )


def _form_line(matches: list[dict[str, Any]]) -> str:
    if not matches:
        return "No completed matches"
    symbols = {"win": "W", "draw": "D", "loss": "L"}
    return " · ".join(symbols[match["outcome"]] for match in reversed(matches))


def _form_table(matches: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": match["match_date"],
                "Venue": match["venue"].title(),
                "Opponent": match["opponent"],
                "Score": f'{match["goals_for"]}–{match["goals_against"]}',
                "Result": match["outcome"].title(),
            }
            for match in matches
        ]
    )


def _render_prediction(prediction: dict[str, Any]) -> None:
    st.subheader("Match forecast")
    columns = st.columns(3)
    labels = ("Home win", "Draw", "Away win")
    keys = (
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
    )
    for column, label, key in zip(columns, labels, keys, strict=True):
        column.metric(label, f'{prediction[key]:.1%}')

    for label, key in zip(labels, keys, strict=True):
        probability = float(prediction[key])
        st.progress(probability, text=f"{label} — {probability:.1%}")
    predicted = RESULT_LABELS[prediction["predicted_result"]]
    st.info(
        f"Highest model probability: **{predicted}**. "
        "This is an educational estimate, not betting advice."
    )


def _render_form(team: str, form: dict[str, Any]) -> None:
    summary = form["summary"]
    st.markdown(f"#### {team}")
    st.caption(f'Oldest → newest: {_form_line(form["matches"])}')
    metric_columns = st.columns(3)
    metric_columns[0].metric("Points", summary["points"])
    metric_columns[1].metric("Goals for", summary["goals_for"])
    metric_columns[2].metric("Goals against", summary["goals_against"])
    st.dataframe(_form_table(form["matches"]), hide_index=True, width="stretch")


def render_dashboard(client: FootCastApiClient | None = None) -> None:
    """Render the dashboard; an injected client keeps the boundary testable."""
    st.set_page_config(
        page_title="FootCast",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("⚽ FootCast")
    st.markdown(
        "Premier League outcome probabilities from a transparent Elo reference model."
    )

    active_client = client or FootCastApiClient(API_URL)
    try:
        if client is None:
            teams_payload, model_info = load_reference_data(API_URL)
        else:
            teams_payload = active_client.teams()
            model_info = active_client.model_info()
    except FootCastApiError as error:
        st.error(str(error))
        st.code("uvicorn footcast.api.main:app --reload")
        st.stop()

    teams = teams_payload["teams"]
    cutoff = date.fromisoformat(str(model_info["data_cutoff"]))
    earliest_prediction = cutoff + timedelta(days=1)
    default_date = max(date.today(), earliest_prediction)

    with st.sidebar:
        st.header("Fixture")
        default_home = teams.index("Arsenal") if "Arsenal" in teams else 0
        home_team = st.selectbox("Home team", teams, index=default_home)
        away_options = [team for team in teams if team != home_team]
        default_away = away_options.index("Chelsea") if "Chelsea" in away_options else 0
        away_team = st.selectbox("Away team", away_options, index=default_away)
        match_date = st.date_input(
            "Match date", value=default_date, min_value=earliest_prediction
        )
        predict_clicked = st.button(
            "Generate forecast", type="primary", width="stretch"
        )
        st.divider()
        st.caption(f"Model: {model_info['model_version']}")
        st.caption(f"Completed data through {cutoff.isoformat()}")

    if predict_clicked:
        try:
            st.session_state["prediction"] = active_client.predict(
                home_team, away_team, match_date.isoformat()
            )
        except FootCastApiError as error:
            st.error(str(error))

    prediction = st.session_state.get("prediction")
    if prediction and (
        prediction["home_team"] != home_team
        or prediction["away_team"] != away_team
        or prediction["match_date"] != match_date.isoformat()
    ):
        prediction = None

    if prediction:
        _render_prediction(prediction)
    else:
        st.info("Choose a fixture and generate a forecast to see probabilities.")

    try:
        if client is None:
            comparison, meetings = load_analytics(API_URL, home_team, away_team)
        else:
            comparison = active_client.compare(home_team, away_team, limit=5)
            meetings = active_client.head_to_head(home_team, away_team, limit=10)
    except FootCastApiError as error:
        st.warning(f"Historical analytics could not be loaded: {error}")
        return

    st.divider()
    st.subheader("Team context")
    elo_columns = st.columns(3)
    elo_columns[0].metric(f"{home_team} Elo", f'{comparison["home_elo"]:.0f}')
    elo_columns[1].metric("Rating difference", f'{comparison["elo_difference"]:+.0f}')
    elo_columns[2].metric(f"{away_team} Elo", f'{comparison["away_elo"]:.0f}')

    form_columns = st.columns(2)
    with form_columns[0]:
        _render_form(home_team, comparison["home"])
    with form_columns[1]:
        _render_form(away_team, comparison["away"])

    st.subheader("Recent head-to-head")
    if meetings["matches"]:
        table = pd.DataFrame(
            [
                {
                    "Date": match["match_date"],
                    "Home": match["home_team"],
                    "Score": f'{match["home_goals"]}–{match["away_goals"]}',
                    "Away": match["away_team"],
                }
                for match in meetings["matches"]
            ]
        )
        st.dataframe(table, hide_index=True, width="stretch")
    else:
        st.caption("No meetings are present in the approved history.")

    with st.expander("How to read this forecast"):
        st.write(model_info["intended_use"].capitalize() + ".")
        for limitation in model_info["limitations"]:
            st.markdown(f"- {limitation}")
        st.caption(
            f"Specification: {model_info['specification_sha256'][:12]}… · "
            f"{model_info['completed_matches']:,} completed matches"
        )


def main() -> None:
    render_dashboard()


if __name__ == "__main__":
    main()
