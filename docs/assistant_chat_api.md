# Phase 9 Assistant Chat API

## Scope

Checkpoint 4 implements the backend half of roadmap Step 20. It exposes a
typed, server-side conversation boundary for the assistant client from
checkpoint 3. It does not add the Streamlit chat panel, stream provider tokens,
run the live model benchmark, or enable provider spending.

The deployed API remains healthy when assistant configuration is absent.
`GET /assistant/status` then reports `available: false`, and chat attempts return
HTTP 503 without importing a provider in the browser or exposing a key.

## Endpoints

### `GET /assistant/status`

Returns availability, the policy version, the ten-turn history limit, the
1,000-character message limit, and four suggested questions. It intentionally
does not return credentials, prices, provider configuration, or internal state.

### `POST /assistant/chat`

Request:

```json
{
  "message": "What is Elo rating in simple terms?",
  "session_id": null
}
```

The first successful response creates a random UUID session. Follow-ups send
that ID back. The response includes the answer, configured model, tool-call
count, latency, and display-safe evidence metadata:

```json
{
  "session_id": "00000000-0000-0000-0000-000000000000",
  "answer": "...",
  "model": "configured-server-model",
  "evidence": [
    {
      "tool_name": "get_metric_definition",
      "answer_mode": "explanation",
      "generated_at": "2026-08-01T12:00:00Z",
      "source": "FootCast approved deterministic services",
      "documentation_version": "phase-9-checkpoint-2-v1"
    }
  ],
  "tool_calls": 1,
  "latency_ms": 420
}
```

Operational token and cost fields remain server-log telemetry rather than
browser payload fields.

### `DELETE /assistant/sessions/{session_id}`

Deletes the ephemeral session and returns whether it existed. A subsequent
follow-up with that ID is rejected. The future dashboard reset button will call
this route and clear its local chat display.

## State and privacy

The process stores only alternating user questions and final assistant answers.
It does not store provider reasoning, tool arguments, credentials, raw logs, or
database rows. History is capped at ten user/assistant turns, sessions expire
after one hour of inactivity, and the in-memory store holds at most 1,000
sessions. State is intentionally ephemeral and disappears on an API restart or
Render free-instance sleep.

This checkpoint does not add a cookie or authenticated account. The random
session UUID is a conversation handle, not an identity or authorization token.

## Security and reliability controls

- Strict Pydantic requests reject unknown fields, malformed UUIDs, blank text,
  and messages over 1,000 characters.
- Middleware rejects chat request bodies over 8 KiB before schema validation.
- A fixed-window limiter permits at most 20 requests per client IP and 10 per
  session per minute. HTTP 429 includes `Retry-After`.
- Provider answers are capped at 12,000 characters before being stored.
- Unknown or expired sessions return 404; unavailable configuration returns
  503; assistant/provider protocol failures return a redacted 502.
- Failed first requests remove the newly allocated session.
- In-memory limits are appropriate for the current single free-tier API
  process. A shared store and authenticated user limit are required before
  horizontally scaling or adding accounts.

Safe repeated-query caching is deliberately absent. An answer can depend on
conversation context, provider/prompt version, and evidence timestamps; cache
semantics will be selected only after Phase 10 evaluation establishes what can
be reused without stale or cross-session output.

## Server configuration

The API image now installs the optional `llm` dependency, but construction is
lazy. Both `OPENAI_API_KEY` and `FOOTCAST_LLM_MODEL` must be present in the
server environment before chat becomes available. Partial configuration fails
startup rather than silently running an unknown setup. No value is added to
`render.yaml`, Git, the dashboard image, or browser code by this checkpoint.

## Test contract

Tests use an injected fake assistant and make no provider requests. They cover
status, availability, typed source metadata, follow-up context, the ten-turn
cap, reset and expiration, IP and session limits, request/body validation,
redacted errors, failed-session cleanup, and the disabled environment path.

Checkpoint 5 now adds the Streamlit chat panel, suggested questions, evidence
cards, observed/prediction labels, progressive final-answer rendering, and
reset control. Live benchmark execution and model comparison remain Phase 10
work.
