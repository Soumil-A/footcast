# Phase 9 Server-side Assistant Client

## Purpose and scope

Checkpoint 3 connects the five deterministic tools to a bounded language-model
loop. It adds no public chat endpoint or dashboard tab, selects no permanent
model, and makes no provider request during import, application startup, or the
test suite. A real request is possible only after the optional `llm` dependency,
an explicit model, and a server-side API key are configured.

The implementation follows OpenAI's current
[function-calling flow](https://developers.openai.com/api/docs/guides/function-calling):
send strict function definitions, preserve every provider output item, execute
each returned `function_call`, append a `function_call_output` with the matching
`call_id`, and continue until the provider returns answer text.

## Runtime boundary

`AssistantClient` owns orchestration and depends only on two small interfaces:

- an `AssistantProvider` that returns normalized response turns; and
- an `AssistantToolRegistry` that publishes schemas and executes validated,
  read-only tools.

`OpenAIResponsesProvider` is the first adapter. It uses the Responses API with
parallel function calls disabled, response storage disabled, an explicit
timeout, and SDK retries disabled so the application's bounded retry policy is
the single source of truth. Stateless requests explicitly include encrypted
reasoning content. Opaque response items, including reasoning items, are passed
back on the next request as required by the API contract.

The provider import is lazy. Existing API and dashboard processes do not need
the OpenAI package and cannot accidentally create a client at startup.

## Safety and reliability controls

- At most four tool calls may execute for one user request.
- The prior context is capped at 20 messages, representing ten user/assistant
  turns.
- Unknown tools, malformed JSON, non-object arguments, and schema failures stop
  the loop before unsupported code can run.
- Only transient timeouts, rate limits, connection failures, and provider server
  failures are retried, at most twice with exponential backoff by default.
- Provider and validation failures are converted to short messages that do not
  include upstream payloads, prompts, credentials, or stack details.
- Logs contain only model name, response/tool counts, tool names, token usage,
  optional cost estimate, and latency. Questions and tool payloads are omitted.

The versioned policy in `footcast.assistant.policy` requires tool grounding,
labels predictions as uncertain, treats tool output as untrusted data, refuses
betting advice and secret requests, and requires provenance in tool-backed
answers.

## Configuration

Copy `.env.example` into the hosting platform's private environment settings;
do not commit a populated file.

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Server-side provider credential; required only when constructing the adapter |
| `FOOTCAST_LLM_MODEL` | Explicit model selected after benchmark evaluation |
| `FOOTCAST_LLM_MAX_TOOL_CALLS` | Hard limit from 1 to 4; default 4 |
| `FOOTCAST_LLM_TIMEOUT_SECONDS` | Timeout for one provider response; default 10 |
| `FOOTCAST_LLM_MAX_RETRIES` | Transient retries from 0 to 3; default 2 |
| `FOOTCAST_LLM_INPUT_USD_PER_MILLION_TOKENS` | Optional current input-token price for telemetry |
| `FOOTCAST_LLM_OUTPUT_USD_PER_MILLION_TOKENS` | Optional current output-token price for telemetry |

Pricing is deliberately configuration, not hard-coded project knowledge,
because provider prices can change. If either price is absent, token usage and
latency are still reported and `estimated_cost_usd` remains `None`.

To enable the real adapter in a future chat service:

```bash
python -m pip install -e ".[llm]"
```

The next checkpoint will construct this client inside FastAPI, add rate limiting
and a typed chat endpoint, run the 42-case benchmark against candidate models,
and expose a dashboard chat surface only after the acceptance gates pass.

## Test contract

All checkpoint tests use scripted fake providers. They cover direct answers,
one- and multi-turn tool flow, preservation of output items, strict schemas,
matching call IDs, unknown tools, invalid arguments, call and context limits,
transient retry exhaustion, redacted errors, configurable model/timeouts,
usage/cost/latency telemetry, policy clauses, environment validation, and the
injected OpenAI adapter contract. They never require a key or network access.
