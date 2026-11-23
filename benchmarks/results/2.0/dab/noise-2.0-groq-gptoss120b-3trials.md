# Scorecard — suite `dab` (harness 0.2.0)

- Observed spend: **$0.2339** over 11 calls · ceiling $2.00

## Config `dab_avaloka_native`

- sut: `avaloka_dab` · llm: `groq/openai/gpt-oss-120b` · sampling: `entire_dataset` · env: `dab-local` · agents: `full_swarm` · trials: 3 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **dab:bookreview:1** ⚠️ | 0.00 | 0.00 | 0 | 0.1343 | 112.8 | — | **3/3 ALL** (budget_exceeded:1, tool_error:2) | `operator_rate` |
| dab:bookreview:2 | 0.67 | 1.00 | 0.6667 | 0.0216 | 34.89 | — | — | `operator_rate` |
| dab:bookreview:3 | 0.67 | 1.00 | 0.6667 | 0.01434 | 32.88 | — | — | `operator_rate` |
| dab:yelp:1 | 0.00 | 0.00 | 0 | 0.009789 | 28.52 | — | provider_error:1, tool_error:1 | `operator_rate` |
| **dab:yelp:2** ⚠️ | 0.00 | 0.00 | 0 | 0.008612 | 19.42 | — | **3/3 ALL** (tool_error:3) | `operator_rate` |
| dab:yelp:3 | 0.00 | 0.00 | 0 | 0.04058 | 39.29 | — | — | `operator_rate` |
| dab:yelp:4 | 0.00 | 0.00 | 0 | 0.03377 | 29.9 | — | tool_error:2 | `operator_rate` |
| **dab:yelp:5** ⚠️ | 0.00 | 0.00 | 0 | 0.004245 | 19.42 | — | **3/3 ALL** (tool_error:3) | `operator_rate` |
| dab:yelp:6 | 0.00 | 0.00 | 0 | 0.01832 | 23.06 | — | tool_error:1 | `operator_rate` |
| dab:yelp:7 | 0.00 | 0.00 | 0 | 0.05356 | 48.95 | — | tool_error:1 | `operator_rate` |

> ⚠️  Rows marked **ALL** scored 0.00 because every trial errored — the model was never successfully measured. Do not read those as quality results.

