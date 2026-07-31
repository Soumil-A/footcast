# Phase 7 Dashboard Contract

## Purpose

The Streamlit dashboard makes FootCast's limited educational reference model
usable as a portfolio product. It displays a future-fixture forecast alongside
historical context without presenting the result as betting guidance.

## Boundary

The dashboard does not import inference, analytics, feature, or data modules.
`footcast.dashboard.client` sends HTTP requests to FastAPI for:

- supported teams and model provenance
- home, draw, and away probabilities
- current Elo rating comparison
- each team's five latest completed matches
- the ten latest head-to-head meetings

This separation means a later model can replace Elo behind the API without
rewriting the page. It also prevents raw files or model internals from moving
into the browser-facing process.

## Run locally

Start the two processes from separate terminals after installing the project:

```bash
uvicorn footcast.api.main:app --reload
```

```bash
streamlit run streamlit_app.py
```

Open `http://127.0.0.1:8501`. To point the dashboard at another API address:

```bash
FOOTCAST_API_URL=http://127.0.0.1:8000 streamlit run streamlit_app.py
```

## Error and cache behavior

- API connection and validation failures become concise user-facing messages.
- Team and model metadata are cached for 60 seconds.
- Analytics responses are cached for 60 seconds by API URL and team matchup.
- Predictions are requested explicitly and retained only in Streamlit session
  state for the matching teams and date.
- Dates on or before the reported completed-data cutoff cannot be selected.

## Test contract

Unit tests verify URL encoding, the exact pre-match prediction payload, HTTP
validation detail, and connection failures. A Streamlit `AppTest` renders the
page headlessly against an injected client. Live verification additionally
starts FastAPI and Streamlit, loads the real approved history, generates an
Arsenal-Chelsea forecast, and checks the rendered form and comparison sections.

## Limitations

- The page is a historical snapshot, not a live results or fixture service.
- Recent form and head-to-head are descriptive; they are not additional inputs
  to this Elo prediction.
- The supported list includes every team observed in the approved history, not
  only clubs in the current Premier League.
- The same model limitations shown by `/model/info` remain in force.
