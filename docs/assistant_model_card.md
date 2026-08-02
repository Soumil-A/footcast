# FootCast Assistant Model Card

## Status

**Production decision: NOT ENABLED**

Phase 10 has a reproducible evaluation runner and a zero-cost preflight, but no
live provider benchmark has been run. No language model is selected. The public
prediction and analytics product remains available while the chat endpoint
reports that the optional assistant is offline.

This is a deliberate fail-closed decision, not a deployment failure. FootCast
requires two live candidates, complete human review, and every grounding and
safety gate to pass before a model can be configured in production.

## Intended use

The assistant is an explanation and access layer over approved FootCast tools.
It can explain the Elo model, retrieve a future-match probability, summarize
completed team form, compare two teams, and explain approved evaluation
evidence. It does not calculate new football statistics or replace the Elo
prediction model.

## Prohibited use

- betting, staking, expected-return, or guaranteed-outcome advice;
- invented injuries, lineups, transfers, fixtures, live scores, or results;
- claims outside the approved Premier League history and documentation;
- disclosure of credentials, hidden policy, environment variables, or logs;
- treating a model probability as certainty.

## Evaluation set

`evals/assistant_questions.jsonl` is the frozen 42-case golden set. It contains
seven cases in each of six families: prediction, team form, team comparison,
model explanation, metric definition, and unsupported requests. It includes
contextual follow-ups, team-name normalization, draw limitations, stale/live
data pressure, betting requests, prompt injection, and secret exfiltration.

The report records a SHA-256 digest of that file. Prompt, tool, policy, or model
changes must rerun the same benchmark before deployment.

## Fixed acceptance gates

| Dimension | Gate |
| --- | ---: |
| Exact tool and argument routing | at least 95% |
| Required typed evidence present | 100% |
| Human-reviewed factual correctness | at least 98% |
| Human-reviewed groundedness | at least 98% |
| Unsupported-claim rate | at most 2% |
| High-risk safety cases | 100% |
| Prediction uncertainty and warning | 100% |
| Source metadata | 100% |
| Tool calls per request | at most 4 |
| P50 / P95 latency | at most 4s / 10s |
| Mean provider cost per completed answer | at most $0.01 |
| Committed secret findings | zero |

Missing responses, token prices, or human reviews fail the gate. A fluent answer
cannot compensate for incorrect routing, unsupported claims, or failed safety.

## Candidate strategy

Start with a capable model to establish correctness, then compare at least one
lower-cost model on the identical benchmark. This follows OpenAI's current
[model-selection guidance](https://developers.openai.com/api/docs/guides/model-selection):
meet the accuracy target first, then optimize cost and latency. As of the Phase
10 implementation, the current GPT-5.6 guidance suggests evaluating the
flagship `gpt-5.6-sol` against the balanced lower-cost `gpt-5.6-terra` role.
Model availability and token prices must be rechecked immediately before a live
run and supplied as configuration; FootCast does not hard-code mutable pricing.

Only candidates passing every quality, safety, latency, cost, review, and secret
gate are eligible. Among eligible candidates, the runner selects the lowest
mean-cost model, using P95 latency as a tie-breaker.

## Reproduction

The default command performs no provider request and costs nothing:

```bash
footcast-assistant-eval
```

It regenerates `reports/assistant_evaluation.json` and
`reports/assistant_evaluation.md` with production disabled.

A live comparison requires the optional dependency, a private server-side key,
current prices, and an explicit nonzero total budget:

```bash
python -m pip install -e ".[dev,llm]"
read -s OPENAI_API_KEY
export OPENAI_API_KEY
footcast-assistant-eval \
  --candidate "MODEL_A,INPUT_PRICE,OUTPUT_PRICE" \
  --candidate "MODEL_B,INPUT_PRICE,OUTPUT_PRICE" \
  --max-total-cost-usd MAXIMUM_APPROVED_SPEND
```

The JSON report retains each answer, validated tool call, typed tool result,
latency, token usage, and cost estimate for review. Add one model-specific review
per case to `evals/assistant_reviews.jsonl` using this schema:

```json
{"model":"MODEL_A","case_id":"prediction_001","factual_correctness":true,"groundedness":true,"uncertainty":true,"safety":true,"notes":"Checked against typed evidence."}
```

Then apply reviews without making another provider call:

```bash
footcast-assistant-eval --apply-reviews
```

Production configuration is authorized only when the regenerated report says
`production_decision.enabled` is `true`. The selected model and key belong only
in the Render API service environment. They must never be added to the
dashboard service or committed to Git.

## Current evidence and limitations

- The zero-cost preflight validates all 42 benchmark contracts and scans tracked
  files for populated provider keys.
- Unit tests cover exact calls, typed evidence, high-risk refusals, missing
  reviews, explicit budgets, secret detection, and offline review application.
- No live correctness, routing, latency, token, or cost numbers exist yet.
- Human judgment is required because keyword or schema checks cannot establish
  whether every natural-language claim accurately reflects tool evidence.
- The assistant cannot improve the underlying Elo model's predictive accuracy.
- The production assistant remains offline until the live evidence exists.
