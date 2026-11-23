# Scorecard — suite `dab` (harness 0.2.0)

- Observed spend: **$1.9816** over 54 calls · ceiling $3.00

## Config `dab_avaloka_native`

- sut: `avaloka_dab` · llm: `openrouter/anthropic/claude-haiku-5.5` · sampling: `entire_dataset` · env: `dab-local` · agents: `full_swarm` · trials: 1 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dab:agnews:1 | 0.00 | 0.00 | 0 | 0.006195 | 36.56 | — | — | `operator_rate` |
| dab:agnews:2 | 0.00 | 0.00 | 0 | 0.01766 | 85.17 | — | — | `operator_rate` |
| dab:agnews:3 | 0.00 | 0.00 | 0 | 0.1615 | 94.3 | — | — | `operator_rate` |
| dab:agnews:4 | 1.00 | 1.00 | 1 | 0.09361 | 70.87 | — | — | `operator_rate` |
| dab:bookreview:1 | 1.00 | 1.00 | 1 | 0.0279 | 39.92 | — | — | `operator_rate` |
| dab:bookreview:2 | 0.00 | 0.00 | 0 | 0.009617 | 33.96 | — | — | `operator_rate` |
| dab:bookreview:3 | 1.00 | 1.00 | 1 | 0.009851 | 33.91 | — | — | `operator_rate` |
| dab:crmarenapro:1 | 1.00 | 1.00 | 1 | 0.006066 | 27.42 | — | — | `operator_rate` |
| dab:crmarenapro:2 | 0.00 | 0.00 | 0 | 0.03418 | 53.88 | — | — | `operator_rate` |
| **dab:crmarenapro:3** ⚠️ | 0.00 | 0.00 | 0 | 0.002905 | 20.41 | — | **1/1 ALL** (answer_error:1) | `operator_rate` |
| dab:crmarenapro:4 | 1.00 | 1.00 | 1 | 0.01023 | 43.83 | — | — | `operator_rate` |
| dab:crmarenapro:5 | 1.00 | 1.00 | 1 | 0.008685 | 34.49 | — | — | `operator_rate` |
| dab:crmarenapro:6 | 1.00 | 1.00 | 1 | 0.02944 | 34.17 | — | — | `operator_rate` |
| dab:crmarenapro:7 | 1.00 | 1.00 | 1 | 0.07461 | 65.84 | — | — | `operator_rate` |
| dab:crmarenapro:8 | 1.00 | 1.00 | 1 | 0.02894 | 56.29 | — | — | `operator_rate` |
| **dab:crmarenapro:9** ⚠️ | 0.00 | 0.00 | 0 | 0.01469 | 40.99 | — | **1/1 ALL** (answer_error:1) | `operator_rate` |
| dab:crmarenapro:10 | 1.00 | 1.00 | 1 | 0.04332 | 50.39 | — | — | `operator_rate` |
| dab:crmarenapro:11 | 1.00 | 1.00 | 1 | 0.00851 | 32.76 | — | — | `operator_rate` |
| dab:crmarenapro:12 | 0.00 | 0.00 | 0 | 0.006185 | 31.82 | — | — | `operator_rate` |
| dab:crmarenapro:13 | 1.00 | 1.00 | 1 | 0.02296 | 42.48 | — | — | `operator_rate` |
| **dab:DEPS_DEV_V1:1** ⚠️ | 0.00 | 0.00 | 0 | 0.06583 | 90.21 | — | **1/1 ALL** (answer_error:1) | `operator_rate` |
| **dab:DEPS_DEV_V1:2** ⚠️ | 0.00 | 0.00 | 0 | 0.2103 | 154.9 | — | **1/1 ALL** (budget_exceeded:1) | `operator_rate` |
| dab:GITHUB_REPOS:1 | 0.00 | 0.00 | 0 | 0.009479 | 62.77 | — | — | `operator_rate` |
| dab:GITHUB_REPOS:2 | 1.00 | 1.00 | 1 | 0.01297 | 48.91 | — | — | `operator_rate` |
| dab:GITHUB_REPOS:3 | 1.00 | 1.00 | 1 | 0.06266 | 52.72 | — | — | `operator_rate` |
| dab:GITHUB_REPOS:4 | 1.00 | 1.00 | 1 | 0.01911 | 35.45 | — | — | `operator_rate` |
| dab:googlelocal:1 | 1.00 | 1.00 | 1 | 0.004219 | 28.7 | — | — | `operator_rate` |
| dab:googlelocal:2 | 0.00 | 0.00 | 0 | 0.003122 | 24.35 | — | — | `operator_rate` |
| dab:googlelocal:3 | 0.00 | 0.00 | 0 | 0.01122 | 44.53 | — | — | `operator_rate` |
| dab:googlelocal:4 | 1.00 | 1.00 | 1 | 0.002706 | 22.42 | — | — | `operator_rate` |
| dab:music_brainz_20k:1 | 0.00 | 0.00 | 0 | 0.002639 | 23.6 | — | — | `operator_rate` |
| dab:music_brainz_20k:2 | 1.00 | 1.00 | 1 | 0.002694 | 23.97 | — | — | `operator_rate` |
| dab:music_brainz_20k:3 | 0.00 | 0.00 | 0 | 0.0541 | 46.81 | — | — | `operator_rate` |
| dab:PANCANCER_ATLAS:1 | 0.00 | 0.00 | 0 | 0.09092 | 69.79 | — | — | `operator_rate` |
| dab:PANCANCER_ATLAS:2 | 1.00 | 1.00 | 1 | 0.06713 | 56.22 | — | — | `operator_rate` |
| dab:PANCANCER_ATLAS:3 | 1.00 | 1.00 | 1 | 0.05602 | 72.43 | — | — | `operator_rate` |
| dab:PATENTS:1 | 0.00 | 0.00 | 0 | 0.09278 | 111.8 | — | — | `operator_rate` |
| dab:PATENTS:2 | 0.00 | 0.00 | 0 | 0.1064 | 90.31 | — | — | `operator_rate` |
| **dab:PATENTS:3** ⚠️ | 0.00 | 0.00 | 0 | 0.03295 | 51.48 | — | **1/1 ALL** (answer_error:1) | `operator_rate` |
| dab:stockindex:1 | 1.00 | 1.00 | 1 | 0.006032 | 31.14 | — | — | `operator_rate` |
| dab:stockindex:2 | 1.00 | 1.00 | 1 | 0.02816 | 35.66 | — | — | `operator_rate` |
| dab:stockindex:3 | 1.00 | 1.00 | 1 | 0.03971 | 68.73 | — | — | `operator_rate` |
| dab:stockmarket:1 | 1.00 | 1.00 | 1 | 0.002543 | 22.17 | — | — | `operator_rate` |
| **dab:stockmarket:2** ⚠️ | 0.00 | 0.00 | 0 | 0.103 | 193.1 | — | **1/1 ALL** (tool_error:1) | `operator_rate` |
| dab:stockmarket:3 | 1.00 | 1.00 | 1 | 0.01622 | 46.67 | — | — | `operator_rate` |
| dab:stockmarket:4 | 0.00 | 0.00 | 0 | 0.0583 | 101.3 | — | — | `operator_rate` |
| dab:stockmarket:5 | 1.00 | 1.00 | 1 | 0.05852 | 119.2 | — | — | `operator_rate` |
| dab:yelp:1 | 1.00 | 1.00 | 1 | 0.03736 | 46.3 | — | — | `operator_rate` |
| dab:yelp:2 | 1.00 | 1.00 | 1 | 0.02189 | 58.72 | — | — | `operator_rate` |
| dab:yelp:3 | 0.00 | 0.00 | 0 | 0.0281 | 46 | — | — | `operator_rate` |
| dab:yelp:4 | 0.00 | 0.00 | 0 | 0.01619 | 57.29 | — | — | `operator_rate` |
| dab:yelp:5 | 0.00 | 0.00 | 0 | 0.005608 | 32.76 | — | — | `operator_rate` |
| dab:yelp:6 | 1.00 | 1.00 | 1 | 0.01465 | 38.17 | — | — | `operator_rate` |
| dab:yelp:7 | 0.00 | 0.00 | 0 | 0.02102 | 73.67 | — | — | `operator_rate` |

> ⚠️  Rows marked **ALL** scored 0.00 because every trial errored — the model was never successfully measured. Do not read those as quality results.

