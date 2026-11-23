# Scorecard — suite `dab` (harness 0.2.0)

- Observed spend: **$0.0000** over 0 calls · ceiling $2.00

## Config `dab_avaloka_native`

- sut: `avaloka_dab` · llm: `gemini-4-argon` · sampling: `entire_dataset` · env: `dab-local` · agents: `full_swarm` · trials: 1 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dab:agnews:1 | 0.00 | 0.00 | 0 | 0.01519 | 89.76 | — | — | `operator_rate` |
| **dab:agnews:2** ⚠️ | 0.00 | 0.00 | 0 | 0.07415 | 125.5 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |
| **dab:agnews:3** ⚠️ | 0.00 | 0.00 | 0 | 0.04496 | 113.5 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |
| dab:agnews:4 | 0.00 | 0.00 | 0 | 0.03745 | 79.34 | — | — | `operator_rate` |
| **dab:bookreview:1** ⚠️ | 0.00 | 0.00 | 0 | 0.01759 | 58.85 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |
| dab:bookreview:2 | 0.00 | 0.00 | 0 | 0.01623 | 25.42 | — | — | `operator_rate` |
| dab:bookreview:3 | 0.00 | 0.00 | 0 | 0.0173 | 73.42 | — | — | `operator_rate` |
| **dab:crmarenapro:1** ⚠️ | 0.00 | 0.00 | 0 | 0.01031 | 141.2 | — | **1/1 ALL** (answer_error:1) | `operator_rate` |
| **dab:crmarenapro:2** ⚠️ | 0.00 | 0.00 | 0 | 0.0037 | 106.9 | — | **1/1 ALL** (answer_error:1) | `operator_rate` |
| dab:crmarenapro:3 | 0.00 | 0.00 | 0 | 0.03495 | 92.45 | — | — | `operator_rate` |
| dab:crmarenapro:4 | 1.00 | 1.00 | 1 | 0.01171 | 22.89 | — | — | `operator_rate` |
| dab:crmarenapro:5 | 0.00 | 0.00 | 0 | 0.01519 | 67.14 | — | — | `operator_rate` |
| **dab:crmarenapro:6** ⚠️ | 0.00 | 0.00 | 0 | 0.0451 | 54.53 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |
| **dab:crmarenapro:7** ⚠️ | 0.00 | 0.00 | 0 | 0.003626 | 122.6 | — | **1/1 ALL** (answer_error:1) | `operator_rate` |
| dab:crmarenapro:8 | 0.00 | 0.00 | 0 | 0.02169 | 62.81 | — | — | `operator_rate` |
| dab:crmarenapro:9 | 0.00 | 0.00 | 0 | 0.01504 | 66.87 | — | — | `operator_rate` |
| **dab:crmarenapro:10** ⚠️ | 0.00 | 0.00 | 0 | 0.02271 | 65.8 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |
| dab:crmarenapro:11 | 1.00 | 1.00 | 1 | 0.02246 | 34.4 | — | — | `operator_rate` |
| dab:crmarenapro:12 | 0.00 | 0.00 | 0 | 0.01009 | 57.72 | — | — | `operator_rate` |
| **dab:crmarenapro:13** ⚠️ | 0.00 | 0.00 | 0 | 0.02098 | 71.08 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |
| dab:DEPS_DEV_V1:1 | 0.00 | 0.00 | 0 | 0.02988 | 48.55 | — | — | `operator_rate` |
| dab:DEPS_DEV_V1:2 | 0.00 | 0.00 | 0 | 0.0298 | 40.42 | — | — | `operator_rate` |
| **dab:GITHUB_REPOS:1** ⚠️ | 0.00 | 0.00 | 0 | 0.01857 | 238 | — | **1/1 ALL** (timeout:1) | `operator_rate` |
| dab:GITHUB_REPOS:2 | 1.00 | 1.00 | 1 | 0.01811 | 40.35 | — | — | `operator_rate` |
| dab:GITHUB_REPOS:3 | 1.00 | 1.00 | 1 | 0.009124 | 52.57 | — | — | `operator_rate` |
| dab:GITHUB_REPOS:4 | 1.00 | 1.00 | 1 | 0.02755 | 100.8 | — | — | `operator_rate` |
| dab:googlelocal:1 | 1.00 | 1.00 | 1 | 0.01378 | 18.22 | — | — | `operator_rate` |
| dab:googlelocal:2 | 0.00 | 0.00 | 0 | 0.0182 | 27.4 | — | — | `operator_rate` |
| dab:googlelocal:3 | 0.00 | 0.00 | 0 | 0.02059 | 26.39 | — | — | `operator_rate` |
| dab:googlelocal:4 | 1.00 | 1.00 | 1 | 0.01544 | 24.63 | — | — | `operator_rate` |
| dab:music_brainz_20k:1 | 0.00 | 0.00 | 0 | 0.009738 | 11.97 | — | — | `operator_rate` |
| dab:music_brainz_20k:2 | 1.00 | 1.00 | 1 | 0.0122 | 35.55 | — | — | `operator_rate` |
| dab:music_brainz_20k:3 | 0.00 | 0.00 | 0 | 0.01169 | 24.96 | — | — | `operator_rate` |
| **dab:PANCANCER_ATLAS:1** ⚠️ | 0.00 | 0.00 | 0 | 0.01448 | 20.44 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |
| **dab:PANCANCER_ATLAS:2** ⚠️ | 0.00 | 0.00 | 0 | 0.01745 | 28.84 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |
| **dab:PANCANCER_ATLAS:3** ⚠️ | 0.00 | 0.00 | 0 | 0.01651 | 46.8 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |
| dab:PATENTS:1 | 0.00 | 0.00 | 0 | 0.02226 | 94.55 | — | — | `operator_rate` |
| **dab:PATENTS:2** ⚠️ | 0.00 | 0.00 | 0 | 0.003132 | 125.7 | — | **1/1 ALL** (answer_error:1) | `operator_rate` |
| **dab:PATENTS:3** ⚠️ | 0.00 | 0.00 | 0 | 0.003108 | 111.6 | — | **1/1 ALL** (answer_error:1) | `operator_rate` |
| dab:stockindex:1 | 0.00 | 0.00 | 0 | 0.01822 | 32.47 | — | — | `operator_rate` |
| dab:stockindex:2 | 0.00 | 0.00 | 0 | 0.01607 | 35.25 | — | — | `operator_rate` |
| dab:stockindex:3 | 0.00 | 0.00 | 0 | 0.02158 | 47.52 | — | — | `operator_rate` |
| dab:stockmarket:1 | 1.00 | 1.00 | 1 | 0.008018 | 12.5 | — | — | `operator_rate` |
| dab:stockmarket:2 | 1.00 | 1.00 | 1 | 0.01251 | 41.51 | — | — | `operator_rate` |
| dab:stockmarket:3 | 0.00 | 0.00 | 0 | 0.01518 | 19.47 | — | — | `operator_rate` |
| dab:stockmarket:4 | 1.00 | 1.00 | 1 | 0.01516 | 20.28 | — | — | `operator_rate` |
| dab:stockmarket:5 | 1.00 | 1.00 | 1 | 0.0133 | 40.1 | — | — | `operator_rate` |
| dab:yelp:1 | 0.00 | 0.00 | 0 | 0.01329 | 193.4 | — | — | `operator_rate` |
| dab:yelp:2 | 0.00 | 0.00 | 0 | 0.02762 | 80.11 | — | — | `operator_rate` |
| dab:yelp:3 | 0.00 | 0.00 | 0 | 0.01139 | 19.04 | — | — | `operator_rate` |
| **dab:yelp:4** ⚠️ | 0.00 | 0.00 | 0 | 0.002296 | 116.4 | — | **1/1 ALL** (answer_error:1) | `operator_rate` |
| dab:yelp:5 | 0.00 | 0.00 | 0 | 0.0262 | 107.9 | — | — | `operator_rate` |
| **dab:yelp:6** ⚠️ | 0.00 | 0.00 | 0 | 0.01644 | 58.7 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |
| **dab:yelp:7** ⚠️ | 0.00 | 0.00 | 0 | 0.01509 | 105.6 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |

> ⚠️  Rows marked **ALL** scored 0.00 because every trial errored — the model was never successfully measured. Do not read those as quality results.

