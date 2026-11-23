# Scorecard — suite `dab` (harness 0.2.0)

- Observed spend: **$0.0031** over 3 calls · ceiling $1.00

## Config `dab_avaloka_native`

- sut: `avaloka_dab` · llm: `groq/openai/gpt-oss-120b` · sampling: `entire_dataset` · env: `dab-local` · agents: `full_swarm` · trials: 1 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dab:bookreview:1 | 0.00 | 0.00 | 0 | 0.0008118 | 8.143 | — | — | `operator_rate` |
| dab:bookreview:2 | 0.00 | 0.00 | 0 | 0.001442 | 8.757 | — | — | `operator_rate` |
| dab:bookreview:3 | 0.00 | 0.00 | 0 | 0.0008509 | 5.608 | — | — | `operator_rate` |

