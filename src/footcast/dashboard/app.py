# ruff: noqa: E501
"""Portfolio-facing Streamlit dashboard backed only by the FootCast API."""

from __future__ import annotations

import os
from datetime import date, timedelta
from html import escape
from typing import Any

import pandas as pd
import streamlit as st

from footcast.dashboard.client import FootCastApiClient, FootCastApiError
from footcast.dashboard.styles import APP_CSS

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
        client.compare(home_team, away_team, limit=10),
        client.head_to_head(home_team, away_team, limit=10),
    )


@st.cache_data(ttl=300, show_spinner=False)
def load_portfolio(api_url: str) -> dict[str, Any]:
    return FootCastApiClient(api_url).portfolio()


def _team_initials(team: str) -> str:
    words = [word for word in team.split() if word]
    if len(words) >= 2:
        return "".join(word[0] for word in words[:2]).upper()
    return team[:2].upper()


def _render_hero(model_info: dict[str, Any], cutoff: date) -> None:
    version = escape(str(model_info["model_version"]))
    st.markdown(
        f"""
        <header class="fc-hero">
          <div>
            <div class="fc-eyebrow">Premier League intelligence</div>
            <h1 class="fc-title">FootCast</h1>
            <p class="fc-subtitle">
              Transparent match probabilities, team momentum, and historical
              context—built on a reproducible Elo reference model.
            </p>
          </div>
          <div class="fc-live" title="{version}">
            <span class="fc-live-dot"></span>
            Model online · {cutoff.isoformat()}
          </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def _render_matchup(home_team: str, away_team: str) -> None:
    home = escape(home_team)
    away = escape(away_team)
    st.markdown(
        f"""
        <section class="fc-matchup" aria-label="Selected fixture">
          <div class="fc-team">
            <div class="fc-team-orb">{escape(_team_initials(home_team))}</div>
            <div>
              <div class="fc-team-role">Home</div>
              <div class="fc-team-name">{home}</div>
            </div>
          </div>
          <div class="fc-vs">VS</div>
          <div class="fc-team fc-team-away">
            <div class="fc-team-orb">{escape(_team_initials(away_team))}</div>
            <div>
              <div class="fc-team-role">Away</div>
              <div class="fc-team-name">{away}</div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_prediction(prediction: dict[str, Any]) -> None:
    labels = ("Home win", "Draw", "Away win")
    result_keys = ("home_win", "draw", "away_win")
    probability_keys = (
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
    )
    probabilities = [float(prediction[key]) for key in probability_keys]
    predicted_result = str(prediction["predicted_result"])

    cards = []
    for label, result_key, probability in zip(
        labels, result_keys, probabilities, strict=True
    ):
        leading = " is-leading" if result_key == predicted_result else ""
        cards.append(
            f'<div class="fc-prob-card{leading}">'
            f'<span class="fc-prob-label">{label}</span>'
            f'<span class="fc-prob-value">{probability:.1%}</span>'
            "</div>"
        )

    segments = "".join(
        f'<span style="width:{probability * 100:.4f}%"></span>'
        for probability in probabilities
    )
    predicted = escape(RESULT_LABELS[predicted_result])
    st.markdown('<div class="fc-section-label">Match forecast</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <section aria-label="Three-way match probabilities">
          <div class="fc-prob-grid">{''.join(cards)}</div>
          <div class="fc-prob-track" aria-hidden="true">{segments}</div>
          <div class="fc-forecast-note">
            <span>Highest model probability · <strong>{predicted}</strong></span>
            <span>Educational estimate · not betting advice</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_empty_forecast() -> None:
    st.markdown('<div class="fc-section-label">Match forecast</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="fc-empty">
          <strong>Forecast engine standing by</strong>
          <span>Select a fixture in the control deck and generate a forecast.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _form_pills(matches: list[dict[str, Any]]) -> str:
    if not matches:
        return '<span class="fc-sidebar-caption">No completed matches</span>'
    labels = {"win": "W", "draw": "D", "loss": "L"}
    return "".join(
        (
            f'<span class="fc-form-pill {escape(str(match["outcome"]))}" '
            f'title="{escape(str(match["outcome"])).title()}">'
            f'{labels[str(match["outcome"])]}</span>'
        )
        for match in reversed(matches)
    )


def _history_rows(matches: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for match in matches:
        venue = "HOME" if match["venue"] == "home" else "AWAY"
        outcome = escape(str(match["outcome"]))
        opponent = escape(str(match["opponent"]))
        score = f'{int(match["goals_for"])}–{int(match["goals_against"])}'
        rows.append(
            '<div class="fc-history-row">'
            f'<span class="fc-venue">{venue}</span>'
            f'<span class="fc-opponent">{opponent}</span>'
            f'<span class="fc-score {outcome}">{score}</span>'
            "</div>"
        )
    return "".join(rows)


def _render_form(team: str, form: dict[str, Any]) -> None:
    summary = form["summary"]
    st.markdown(f"#### {escape(team)}")
    st.markdown(
        f'<div class="fc-form-pills">{_form_pills(form["matches"])}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="fc-mini-stats">
          <div class="fc-mini-stat"><span>Points</span><strong>{int(summary['points'])}</strong></div>
          <div class="fc-mini-stat"><span>Goals for</span><strong>{int(summary['goals_for'])}</strong></div>
          <div class="fc-mini-stat"><span>Against</span><strong>{int(summary['goals_against'])}</strong></div>
        </div>
        <div class="fc-history-list">{_history_rows(form['matches'])}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_elo(home_team: str, away_team: str, comparison: dict[str, Any]) -> None:
    home_elo = float(comparison["home_elo"])
    away_elo = float(comparison["away_elo"])
    difference = float(comparison["elo_difference"])
    marker = min(82.0, max(18.0, 50.0 + difference / 5.0))
    leader = home_team if difference >= 0 else away_team
    st.markdown('<div class="fc-section-label">Strength signal</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <section class="fc-elo-grid" aria-label="Elo rating comparison">
          <div class="fc-stat-card">
            <div class="fc-stat-label">{escape(home_team)} Elo</div>
            <div class="fc-stat-value">{home_elo:.0f}</div>
          </div>
          <div class="fc-stat-card fc-elo-center">
            <div class="fc-stat-label">Rating edge</div>
            <div class="fc-elo-delta">{escape(leader)} · {abs(difference):.0f}</div>
            <div class="fc-elo-track">
              <span class="fc-elo-marker" style="left:{marker:.2f}%"></span>
            </div>
            <div class="fc-elo-caption">Historical rating balance</div>
          </div>
          <div class="fc-stat-card right">
            <div class="fc-stat-label">{escape(away_team)} Elo</div>
            <div class="fc-stat-value">{away_elo:.0f}</div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_head_to_head(meetings: dict[str, Any]) -> None:
    st.markdown('<div class="fc-section-label">Recent head-to-head</div>', unsafe_allow_html=True)
    if not meetings["matches"]:
        st.caption("No meetings are present in the approved history.")
        return
    rows = []
    for match in meetings["matches"]:
        rows.append(
            '<div class="fc-h2h-row">'
            f'<span class="fc-h2h-date">{escape(str(match["match_date"]))}</span>'
            f'<span class="fc-h2h-home">{escape(str(match["home_team"]))}</span>'
            f'<span class="fc-h2h-score">{int(match["home_goals"])}–'
            f'{int(match["away_goals"])}</span>'
            f'<span class="fc-h2h-away">{escape(str(match["away_team"]))}</span>'
            "</div>"
        )
    st.markdown(
        f'<div class="fc-h2h-list">{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


def _team_trend_frame(form: dict[str, Any]) -> pd.DataFrame:
    rows = []
    running_points = 0
    for match in reversed(form["matches"]):
        running_points += int(match["points"])
        rows.append(
            {
                "Match date": str(match["match_date"]),
                "Cumulative points": running_points,
                "Goals for": int(match["goals_for"]),
                "Goals against": int(match["goals_against"]),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["Cumulative points", "Goals for", "Goals against"]
        )
    return pd.DataFrame(rows).set_index("Match date")


def _render_team_analytics(
    home_team: str,
    away_team: str,
    comparison: dict[str, Any],
    meetings: dict[str, Any],
    portfolio: dict[str, Any],
) -> None:
    st.markdown(
        '<div class="fc-section-label">Ten-match performance trends</div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(2, gap="large")
    for column, team, key in zip(
        columns, (home_team, away_team), ("home", "away"), strict=True
    ):
        form = comparison[key]
        summary = form["summary"]
        frame = _team_trend_frame(form)
        with column:
            st.subheader(team)
            metric_columns = st.columns(3)
            metric_columns[0].metric("Points", int(summary["points"]))
            metric_columns[1].metric("Goals for", int(summary["goals_for"]))
            metric_columns[2].metric(
                "Goal difference",
                int(summary["goals_for"]) - int(summary["goals_against"]),
            )
            st.caption("Cumulative points across the displayed matches")
            st.line_chart(frame[["Cumulative points"]], color="#22d3ee")
            st.caption("Goals scored and conceded by match")
            st.bar_chart(
                frame[["Goals for", "Goals against"]],
                color=["#34d399", "#fb7185"],
            )

    st.markdown(
        '<div class="fc-section-label">Head-to-head balance</div>',
        unsafe_allow_html=True,
    )
    outcomes = [match["team_a_outcome"] for match in meetings["matches"]]
    h2h = pd.DataFrame(
        {
            "Matches": [
                outcomes.count("win"),
                outcomes.count("draw"),
                outcomes.count("loss"),
            ]
        },
        index=[f"{home_team} wins", "Draws", f"{away_team} wins"],
    )
    if outcomes:
        st.bar_chart(h2h, color="#8b5cf6")
    else:
        st.caption("No meetings are present in the approved history.")

    st.markdown(
        '<div class="fc-section-label">Approved-history snapshot</div>',
        unsafe_allow_html=True,
    )
    overview = st.columns(4)
    overview[0].metric("Completed matches", f"{portfolio['completed_matches']:,}")
    overview[1].metric("Seasons", int(portfolio["season_count"]))
    overview[2].metric("First match", str(portfolio["first_match_date"]))
    overview[3].metric("Data cutoff", str(portfolio["data_cutoff"]))

    history_columns = st.columns(2, gap="large")
    with history_columns[0]:
        st.subheader("Outcome distribution")
        distribution = pd.DataFrame(portfolio["outcome_distribution"])
        distribution["Outcome"] = distribution["outcome"].map(RESULT_LABELS)
        distribution["Share"] = distribution["share"] * 100
        st.bar_chart(
            distribution.set_index("Outcome")[["Share"]], color="#22d3ee"
        )
        st.caption("Percentage of all approved completed matches")
    with history_columns[1]:
        st.subheader("Current Elo leaders")
        ranking = pd.DataFrame(portfolio["strength_ranking"])
        st.bar_chart(ranking.set_index("team")[["elo"]], color="#8b5cf6")
        st.caption(
            "Final ratings after replaying the approved history; the list can "
            "include clubs outside the current Premier League."
        )


def _render_model_insights(
    model_info: dict[str, Any], portfolio: dict[str, Any]
) -> None:
    st.markdown(
        '<div class="fc-section-label">Untouched final-test evidence</div>',
        unsafe_allow_html=True,
    )
    st.info(portfolio["selection_note"])
    evidence = st.columns(4)
    deployed = next(
        item for item in portfolio["benchmarks"] if item["model"] == "Elo (deployed)"
    )
    evidence[0].metric("Test season", portfolio["test_season"])
    evidence[1].metric("Test matches", int(portfolio["test_matches"]))
    evidence[2].metric("Elo accuracy", f"{deployed['accuracy']:.1%}")
    evidence[3].metric("Elo log loss", f"{deployed['log_loss']:.3f}")

    chart_columns = st.columns(2, gap="large")
    benchmarks = pd.DataFrame(portfolio["benchmarks"]).set_index("model")
    with chart_columns[0]:
        st.subheader("Model comparison")
        st.bar_chart(
            benchmarks[["accuracy", "macro_f1"]],
            color=["#22d3ee", "#8b5cf6"],
        )
        st.caption("Accuracy and macro F1: higher is better")
    with chart_columns[1]:
        st.subheader("Probability quality")
        st.bar_chart(benchmarks[["log_loss"]], color="#f472b6")
        st.caption("Log loss: lower is better")

    diagnostic_columns = st.columns(2, gap="large")
    recall = pd.DataFrame(
        {
            "Recall": [
                portfolio["deployed_elo_recall"][label]
                for label in portfolio["class_order"]
            ]
        },
        index=[RESULT_LABELS[label] for label in portfolio["class_order"]],
    )
    confusion = pd.DataFrame(
        portfolio["deployed_elo_confusion_matrix"],
        index=[f"Actual {RESULT_LABELS[label]}" for label in portfolio["class_order"]],
        columns=[
            f"Predicted {RESULT_LABELS[label]}" for label in portfolio["class_order"]
        ],
    )
    with diagnostic_columns[0]:
        st.subheader("Elo recall by outcome")
        st.bar_chart(recall, color="#fbbf24")
        st.warning(
            "Draw recall is 0%. The deployed reference model assigns a draw "
            "probability, but does not select draws as its highest-probability label."
        )
    with diagnostic_columns[1]:
        st.subheader("Elo confusion matrix")
        st.dataframe(confusion, width="stretch")
        st.caption("Rows are actual results; columns are model selections")

    with st.expander("Model transparency and responsible use"):
        st.write(model_info["intended_use"].capitalize() + ".")
        for limitation in model_info["limitations"]:
            st.markdown(f"- {limitation}")
        st.caption(
            f"Specification: {model_info['specification_sha256'][:12]}… · "
            f"{model_info['completed_matches']:,} completed matches · "
            "Recent form and dashboard charts are descriptive context, not extra inputs."
        )


def render_dashboard(client: FootCastApiClient | None = None) -> None:
    """Render the dashboard; an injected client keeps the boundary testable."""
    st.set_page_config(
        page_title="FootCast · Match Intelligence",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)

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
        st.markdown(
            """
            <div class="fc-sidebar-brand">
              <div class="fc-sidebar-mark">FC</div>
              <div class="fc-sidebar-name">FootCast Control</div>
              <div class="fc-sidebar-caption">Configure a future fixture</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.header("Control deck")
        default_home = teams.index("Arsenal") if "Arsenal" in teams else 0
        home_team = st.selectbox("Home team", teams, index=default_home)
        away_options = [team for team in teams if team != home_team]
        default_away = away_options.index("Chelsea") if "Chelsea" in away_options else 0
        away_team = st.selectbox("Away team", away_options, index=default_away)
        match_date = st.date_input(
            "Match date", value=default_date, min_value=earliest_prediction
        )
        predict_clicked = st.button(
            "Generate forecast →", type="primary", width="stretch"
        )
        st.markdown(
            f"""
            <div class="fc-meta">
              <div class="fc-meta-row"><span>Engine</span><strong>{escape(str(model_info['model_version']))}</strong></div>
              <div class="fc-meta-row"><span>Data cutoff</span><strong>{cutoff.isoformat()}</strong></div>
              <div class="fc-meta-row"><span>Holdout</span><strong>Sealed</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if predict_clicked:
        st.session_state.pop("prediction", None)
        try:
            with st.spinner("Calculating fixture probabilities…"):
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

    try:
        if client is None:
            comparison, meetings = load_analytics(API_URL, home_team, away_team)
            portfolio = load_portfolio(API_URL)
        else:
            comparison = active_client.compare(home_team, away_team, limit=10)
            meetings = active_client.head_to_head(home_team, away_team, limit=10)
            portfolio = active_client.portfolio()
    except FootCastApiError as error:
        st.warning(f"Historical analytics could not be loaded: {error}")
        return

    _render_hero(model_info, cutoff)
    _render_matchup(home_team, away_team)
    forecast_tab, analytics_tab, model_tab = st.tabs(
        ["Match Forecast", "Team Analytics", "Model Insights"]
    )

    with forecast_tab:
        if prediction:
            _render_prediction(prediction)
        else:
            _render_empty_forecast()
        _render_elo(home_team, away_team, comparison)
        st.markdown(
            '<div class="fc-section-label">Momentum monitor · last 10</div>',
            unsafe_allow_html=True,
        )
        form_columns = st.columns(2, gap="large")
        with form_columns[0]:
            with st.container(border=True):
                _render_form(home_team, comparison["home"])
        with form_columns[1]:
            with st.container(border=True):
                _render_form(away_team, comparison["away"])
        _render_head_to_head(meetings)

    with analytics_tab:
        _render_team_analytics(
            home_team, away_team, comparison, meetings, portfolio
        )

    with model_tab:
        _render_model_insights(model_info, portfolio)


def main() -> None:
    render_dashboard()


if __name__ == "__main__":
    main()
