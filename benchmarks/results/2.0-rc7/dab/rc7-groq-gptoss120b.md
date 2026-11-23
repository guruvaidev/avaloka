# Scorecard — suite `dab` (harness 0.2.0)

- Observed spend: **$0.6470** over 22 calls · ceiling $1.50

## Config `dab_avaloka_native`

- sut: `avaloka_dab` · llm: `groq/openai/gpt-oss-120b` · sampling: `entire_dataset` · env: `dab-local` · agents: `full_swarm` · trials: 1 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dab:agnews:1 | 0.00 | 0.00 | 0 | 0.0284 | 24.58 | — | — | `operator_rate` |
| dab:agnews:2 | 0.00 | 0.00 | 0 | 0.01647 | 27.31 | — | — | `operator_rate` |
| dab:agnews:3 | 0.00 | 0.00 | 0 | 0.1402 | 109.2 | — | — | `operator_rate` |
| **dab:agnews:4** ⚠️ | 0.00 | 0.00 | 0 | 0.002268 | 22.66 | — | **1/1 ALL** (provider_error:1) | `operator_rate` |
| **dab:bookreview:1** ⚠️ | 0.00 | 0.00 | 0 | 0.01204 | 21.48 | — | **1/1 ALL** (tool_error:1) | `operator_rate` |
| dab:bookreview:2 | 1.00 | 1.00 | 1 | 0.0112 | 25.81 | — | — | `operator_rate` |
| dab:bookreview:3 | 1.00 | 1.00 | 1 | 0.01125 | 24.95 | — | — | `operator_rate` |
| dab:crmarenapro:1 | 1.00 | 1.00 | 1 | 0.003343 | 20.25 | — | — | `operator_rate` |
| dab:crmarenapro:2 | 1.00 | 1.00 | 1 | 0.02107 | 25.9 | — | — | `operator_rate` |
| dab:crmarenapro:3 | 1.00 | 1.00 | 1 | 0.004296 | 23.46 | — | — | `operator_rate` |
| **dab:crmarenapro:4** ⚠️ | 0.00 | 0.00 | 0 | 0.01467 | 27.85 | — | **1/1 ALL** (provider_error:1) | `operator_rate` |
| dab:crmarenapro:5 | 1.00 | 1.00 | 1 | 0.02189 | 43.07 | — | — | `operator_rate` |
| dab:crmarenapro:6 | 0.00 | 0.00 | 0 | 0.004599 | 18.92 | — | — | `operator_rate` |
| **dab:crmarenapro:7** ⚠️ | 0.00 | 0.00 | 0 | 0.01286 | 31.65 | — | **1/1 ALL** (provider_error:1) | `operator_rate` |
| dab:crmarenapro:8 | 0.00 | 0.00 | 0 | 0.01551 | 29.57 | — | — | `operator_rate` |
| dab:crmarenapro:9 | 1.00 | 1.00 | 1 | 0.01188 | 27.17 | — | — | `operator_rate` |
| **dab:crmarenapro:10** ⚠️ | 0.00 | 0.00 | 0 | 0.0318 | 55.18 | — | **1/1 ALL** (provider_error:1) | `operator_rate` |
| dab:crmarenapro:11 | 0.00 | 0.00 | 0 | 0.01653 | 36.91 | — | — | `operator_rate` |
| dab:crmarenapro:12 | 0.00 | 0.00 | 0 | 0.00342 | 21.63 | — | — | `operator_rate` |
| dab:crmarenapro:13 | 1.00 | 1.00 | 1 | 0.004113 | 20.82 | — | — | `operator_rate` |
| **dab:DEPS_DEV_V1:1** ⚠️ | 0.00 | 0.00 | 0 | 0.02024 | 144.8 | — | **1/1 ALL** (timeout:1) | `operator_rate` |
| **dab:DEPS_DEV_V1:2** ⚠️ | 0.00 | 0.00 | 0 | 0.05426 | 65.29 | — | **1/1 ALL** (provider_error:1) | `operator_rate` |
| **dab:GITHUB_REPOS:1** ⚠️ | 0.00 | 0.00 | 0 | 0.1952 | 208.5 | — | **1/1 ALL** (tool_error:1) | `operator_rate` |
| **dab:GITHUB_REPOS:2** ⚠️ | 0.00 | 0.00 | 0 | 0.01026 | 23.82 | — | **1/1 ALL** (tool_error:1) | `operator_rate` |
| **dab:GITHUB_REPOS:3** ⚠️ | 0.00 | 0.00 | 0 | 0.0605 | 61.06 | — | **1/1 ALL** (provider_error:1) | `operator_rate` |
| **dab:GITHUB_REPOS:4** ⚠️ | 0.00 | 0.00 | 0 | 0.04804 | 99.55 | — | **1/1 ALL** (tool_error:1) | `operator_rate` |
| **dab:googlelocal:1** ⚠️ | 0.00 | 0.00 | 0 | 0.005653 | 26.05 | — | **1/1 ALL** (provider_error:1) | `operator_rate` |
| dab:googlelocal:2 | 0.00 | 0.00 | 0 | 0.00828 | 30.2 | — | — | `operator_rate` |
| dab:googlelocal:3 | 1.00 | 1.00 | 1 | 0.008096 | 28.25 | — | — | `operator_rate` |
| **dab:googlelocal:4** ⚠️ | 0.00 | 0.00 | 0 | 0.003942 | 21.6 | — | **1/1 ALL** (provider_error:1) | `operator_rate` |
| **dab:music_brainz_20k:1** ⚠️ | 0.00 | 0.00 | 0 | 0.00671 | 24.86 | — | **1/1 ALL** (provider_error:1) | `operator_rate` |
| **dab:music_brainz_20k:2** ⚠️ | 0.00 | 0.00 | 0 | 0.001935 | 17.44 | — | **1/1 ALL** (provider_error:1) | `operator_rate` |
| dab:music_brainz_20k:3 | 0.00 | 0.00 | 0 | 0.02196 | 21.93 | — | — | `operator_rate` |
| dab:PANCANCER_ATLAS:1 | 0.00 | 0.00 | 0 | 0.03347 | 35.95 | — | — | `operator_rate` |
| **dab:PANCANCER_ATLAS:2** ⚠️ | 0.00 | 0.00 | 0 | 0.01466 | 29.15 | — | **1/1 ALL** (tool_error:1) | `operator_rate` |
| dab:PANCANCER_ATLAS:3 | 0.00 | 0.00 | 0 | 0.03439 | 45.83 | — | — | `operator_rate` |
| **dab:PATENTS:1** ⚠️ | 0.00 | 0.00 | 0 | 0.03467 | 77.38 | — | **1/1 ALL** (tool_error:1) | `operator_rate` |
| **dab:PATENTS:2** ⚠️ | 0.00 | 0.00 | 0 | 0.01414 | 49.08 | — | **1/1 ALL** (provider_error:1) | `operator_rate` |
| **dab:PATENTS:3** ⚠️ | 0.00 | 0.00 | 0 | 0.09964 | 142.3 | — | **1/1 ALL** (provider_error:1) | `operator_rate` |
| dab:stockindex:1 | 0.00 | 0.00 | 0 | 0.02979 | 30.62 | — | — | `operator_rate` |
| dab:stockindex:2 | 1.00 | 1.00 | 1 | 0.03197 | 42.64 | — | — | `operator_rate` |
| dab:stockindex:3 | 0.00 | 0.00 | 0 | 0.03886 | 51.93 | — | — | `operator_rate` |
| dab:stockmarket:1 | 1.00 | 1.00 | 1 | 0.002327 | 17.77 | — | — | `operator_rate` |
| **dab:stockmarket:2** ⚠️ | 0.00 | 0.00 | 0 | 0.06147 | 49.66 | — | **1/1 ALL** (budget_exceeded:1) | `operator_rate` |
| dab:stockmarket:3 | 0.00 | 0.00 | 0 | 0.05273 | 44.99 | — | — | `operator_rate` |
| **dab:stockmarket:4** ⚠️ | 0.00 | 0.00 | 0 | 0.09693 | 87.47 | — | **1/1 ALL** (budget_exceeded:1) | `operator_rate` |
| dab:stockmarket:5 | 0.00 | 0.00 | 0 | 0.04275 | 50.73 | — | — | `operator_rate` |
| **dab:yelp:1** ⚠️ | 0.00 | 0.00 | 0 | 0.008701 | 28.54 | — | **1/1 ALL** (provider_error:1) | `operator_rate` |
| **dab:yelp:2** ⚠️ | 0.00 | 0.00 | 0 | 0.02832 | 26.83 | — | **1/1 ALL** (tool_error:1) | `operator_rate` |
| dab:yelp:3 | 0.00 | 0.00 | 0 | 0.006747 | 25.09 | — | — | `operator_rate` |
| **dab:yelp:4** ⚠️ | 0.00 | 0.00 | 0 | 0.01572 | 22.03 | — | **1/1 ALL** (tool_error:1) | `operator_rate` |
| **dab:yelp:5** ⚠️ | 0.00 | 0.00 | 0 | 0.09499 | 67.93 | — | **1/1 ALL** (tool_error:1) | `operator_rate` |
| dab:yelp:6 | 0.00 | 0.00 | 0 | 0.01764 | 31.24 | — | — | `operator_rate` |
| **dab:yelp:7** ⚠️ | 0.00 | 0.00 | 0 | 0.01975 | 41.75 | — | **1/1 ALL** (tool_error:1) | `operator_rate` |

> ⚠️  Rows marked **ALL** scored 0.00 because every trial errored — the model was never successfully measured. Do not read those as quality results.

