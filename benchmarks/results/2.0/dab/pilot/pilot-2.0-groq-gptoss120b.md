# Scorecard — suite `dab` (harness 0.2.0)

- Observed spend: **$0.0770** over 3 calls · ceiling $2.00

## Config `dab_avaloka_native`

- sut: `avaloka_dab` · llm: `groq/openai/gpt-oss-120b` · sampling: `entire_dataset` · env: `dab-local` · agents: `full_swarm` · trials: 1 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dab:bookreview:1 | 1.00 | 1.00 | 1 | 0.0337 | 30.96 | — | — | `operator_rate` |
| **dab:bookreview:2** ⚠️ | 0.00 | 0.00 | 0 | 0.03047 | 52.09 | — | **1/1 ALL** (provider_error:1) | `operator_rate` |
| dab:bookreview:3 | 1.00 | 1.00 | 1 | 0.01278 | 26.95 | — | — | `operator_rate` |

> ⚠️  Rows marked **ALL** scored 0.00 because every trial errored — the model was never successfully measured. Do not read those as quality results.

