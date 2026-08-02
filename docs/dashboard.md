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
- approved-history coverage and outcome distribution
- current top-ten Elo ranking
- frozen final-test benchmarks and diagnostics

This separation means a later model can replace Elo behind the API without
rewriting the page. It also prevents raw files or model internals from moving
into the browser-facing process.

## Visual system

The Phase 7 polish checkpoint uses a responsive broadcast-analytics direction:

- a deep navy background with restrained cyan and violet signals
- glass-style matchup, probability, Elo, and history surfaces
- text monograms instead of external crest assets or licensing dependencies
- three probability cards plus one proportional spectrum
- color-and-letter W/D/L chips so form does not depend on color alone
- a collapsible control deck on narrow screens
- a model-status pill, visible cutoff, and sealed-holdout indicator

## Information architecture

The portfolio product divides the page into four tabs:

1. **Match Forecast** contains the only interactive prediction surface plus
   Elo, recent form, and head-to-head context.
2. **Team Analytics** charts ten-match cumulative points, goals for and against,
   head-to-head balance, dataset coverage, result distribution, and Elo leaders.
3. **Model Insights** compares the models on the untouched 2024-25 test and
   displays the deployed Elo model's class recall and confusion matrix.
4. **Ask FootCast** provides tool-grounded conversational access with visible
   evidence and clear answer-mode labels.

Native Streamlit charts keep the container dependency set small. Metric names
and captions state whether higher or lower is better, and the zero draw recall
is shown explicitly rather than hidden behind overall accuracy.

The layout collapses its matchup, probability, and Elo grids below `760px` and
does not introduce horizontal scrolling at a `390px` viewport.

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
- Matchup analytics are cached for 60 seconds by API URL and team matchup.
- The immutable portfolio summary is cached for five minutes.
- Predictions are requested explicitly and retained only in Streamlit session
  state for the matching teams and date.
- Dates on or before the reported completed-data cutoff cannot be selected.

## Ask FootCast

Phase 9 checkpoint 5 adds a fourth dashboard workspace for conversational
analytics. The tab calls only the typed API routes documented in
[assistant_chat_api.md](assistant_chat_api.md); it never imports the provider,
assistant tools, inference service, or match data.

The first view offers four server-provided suggested questions and a 1,000
character chat input. Successful answers render progressively inside Streamlit
after the API response arrives and include:

- a visible `Model prediction`, `Observed history`, `Approved explanation`, or
  `Assistant guidance` label;
- the final grounded answer;
- one evidence card per executed tool with its source and generation timestamp;
- available cutoff, model/documentation version, season, window, and sample
  size metadata; and
- the language model identifier, tool-call count, and end-to-end API latency.

The reset control asks the API to delete the ephemeral session, then clears the
dashboard transcript. Suggested questions return after reset. No transcript is
stored in Streamlit's shared data cache.

Availability is fetched separately from deterministic analytics. If the
assistant endpoint is unavailable, missing during a rolling deploy, or not
configured, forecasts and analytics still render. The chat tab shows a clear
offline panel explaining that the key belongs only on the API server.

This is not yet provider-token streaming: the secured API completes and
validates the tool loop before the dashboard progressively renders the final
text. End-to-end streaming should be considered only after Phase 10 evaluation
confirms that partial responses cannot bypass grounding and evidence display.

## Test contract

Unit tests verify URL encoding, assistant routes, isolated request timeouts, the
portfolio endpoint, the exact pre-match prediction payload, HTTP validation
detail, and connection failures. A Streamlit `AppTest` renders the page
headlessly against an injected client and exercises forecast state, typed chat,
evidence display, suggested questions, reset, and unavailable behavior. Live
verification additionally starts FastAPI and Streamlit, loads the real approved
history, generates an Arsenal-Chelsea forecast, and checks the rendered form
and comparison sections at desktop and mobile widths.

## Limitations

- The page is a historical snapshot, not a live results or fixture service.
- Recent form and head-to-head are descriptive; they are not additional inputs
  to this Elo prediction.
- Model charts report frozen 2024-25 test evidence; they are not live accuracy.
- The supported list includes every team observed in the approved history, not
  only clubs in the current Premier League.
- The same model limitations shown by `/model/info` remain in force.
