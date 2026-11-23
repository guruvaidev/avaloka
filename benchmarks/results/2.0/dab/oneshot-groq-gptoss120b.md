# Scorecard — suite `dab` (harness 0.2.0)

- Observed spend: **$0.0000** over 0 calls · ceiling $1.00

## Config `dab_avaloka_native`

- sut: `avaloka_dab` · llm: `groq/openai/gpt-oss-120b` · sampling: `entire_dataset` · env: `dab-local` · agents: `full_swarm` · trials: 1 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dab:agnews:1 | 0.00 | 0.00 | 0 | 0.0005091 | 29.54 | — | — | `operator_rate` |
| dab:agnews:2 | 0.00 | 0.00 | 0 | 0.0008271 | 14.06 | — | — | `operator_rate` |
| dab:agnews:3 | 0.00 | 0.00 | 0 | 0.0008971 | 22.9 | — | — | `operator_rate` |
| dab:agnews:4 | 0.00 | 0.00 | 0 | 0.0008525 | 15.11 | — | — | `operator_rate` |
| dab:bookreview:1 | 0.00 | 0.00 | 0 | 0.001322 | 18.52 | — | — | `operator_rate` |
| dab:bookreview:2 | 0.00 | 0.00 | 0 | 0.0008214 | 14.6 | — | — | `operator_rate` |
| dab:bookreview:3 | 0.00 | 0.00 | 0 | 0.001192 | 14.91 | — | — | `operator_rate` |
| dab:crmarenapro:1 | 1.00 | 1.00 | 1 | 0.0008742 | 12.49 | — | — | `operator_rate` |
| **dab:crmarenapro:2** ⚠️ | 0.00 | 0.00 | 0 | 0.00119 | 10.73 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |
| **dab:crmarenapro:3** ⚠️ | 0.00 | 0.00 | 0 | 0.001067 | 13.11 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |
| **dab:crmarenapro:4** ⚠️ | 0.00 | 0.00 | 0 | 0.001277 | 13.65 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |
| dab:crmarenapro:5 | 0.00 | 0.00 | 0 | 0.001082 | 12.86 | — | — | `operator_rate` |
| **dab:crmarenapro:6** ⚠️ | 0.00 | 0.00 | 0 | 0.001522 | 18.73 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |
| dab:crmarenapro:7 | 0.00 | 0.00 | 0 | 0.000843 | 12.84 | — | — | `operator_rate` |
| **dab:crmarenapro:8** ⚠️ | 0.00 | 0.00 | 0 | 0.00144 | 12.9 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |
| **dab:crmarenapro:9** ⚠️ | 0.00 | 0.00 | 0 | 0.0009401 | 12.98 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |
| **dab:crmarenapro:10** ⚠️ | 0.00 | 0.00 | 0 | 0.001341 | 19.13 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |
| dab:crmarenapro:11 | 0.00 | 0.00 | 0 | 0.001455 | 14.68 | — | — | `operator_rate` |
| **dab:crmarenapro:12** ⚠️ | 0.00 | 0.00 | 0 | 0.001009 | 12.91 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |
| dab:crmarenapro:13 | 1.00 | 1.00 | 1 | 0.0011 | 15.19 | — | — | `operator_rate` |
| dab:DEPS_DEV_V1:1 | 0.00 | 0.00 | 0 | 0.001311 | 34.07 | — | — | `operator_rate` |
| dab:DEPS_DEV_V1:2 | 0.00 | 0.00 | 0 | 0.001481 | 37.83 | — | — | `operator_rate` |
| dab:GITHUB_REPOS:1 | 0.00 | 0.00 | 0 | 0.0009575 | 40.92 | — | — | `operator_rate` |
| dab:GITHUB_REPOS:2 | 0.00 | 0.00 | 0 | 0.001139 | 18.81 | — | — | `operator_rate` |
| dab:GITHUB_REPOS:3 | 1.00 | 1.00 | 1 | 0.0009897 | 33.39 | — | — | `operator_rate` |
| dab:GITHUB_REPOS:4 | 0.00 | 0.00 | 0 | 0.0008199 | 28.78 | — | — | `operator_rate` |
| dab:googlelocal:1 | 0.00 | 0.00 | 0 | 0.001056 | 9.238 | — | — | `operator_rate` |
| dab:googlelocal:2 | 0.00 | 0.00 | 0 | 0.0007428 | 9.191 | — | — | `operator_rate` |
| dab:googlelocal:3 | 0.00 | 0.00 | 0 | 0.001197 | 9.661 | — | — | `operator_rate` |
| dab:googlelocal:4 | 1.00 | 1.00 | 1 | 0.0008764 | 9.459 | — | — | `operator_rate` |
| dab:music_brainz_20k:1 | 0.00 | 0.00 | 0 | 0.0006511 | 7.239 | — | — | `operator_rate` |
| dab:music_brainz_20k:2 | 1.00 | 1.00 | 1 | 0.0008593 | 11.42 | — | — | `operator_rate` |
| dab:music_brainz_20k:3 | 0.00 | 0.00 | 0 | 0.0007413 | 7.718 | — | — | `operator_rate` |
| dab:PANCANCER_ATLAS:1 | 0.00 | 0.00 | 0 | 0.001379 | 10.35 | — | — | `operator_rate` |
| **dab:PANCANCER_ATLAS:2** ⚠️ | 0.00 | 0.00 | 0 | 0.001418 | 11.5 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |
| **dab:PANCANCER_ATLAS:3** ⚠️ | 0.00 | 0.00 | 0 | 0.001213 | 11.8 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |
| dab:PATENTS:1 | 0.00 | 0.00 | 0 | 0.001209 | 36.12 | — | — | `operator_rate` |
| dab:PATENTS:2 | 0.00 | 0.00 | 0 | 0.001869 | 19.26 | — | — | `operator_rate` |
| dab:PATENTS:3 | 0.00 | 0.00 | 0 | 0.001503 | 39.23 | — | — | `operator_rate` |
| dab:stockindex:1 | 0.00 | 0.00 | 0 | 0.001182 | 7.434 | — | — | `operator_rate` |
| dab:stockindex:2 | 0.00 | 0.00 | 0 | 0.001491 | 7.856 | — | — | `operator_rate` |
| dab:stockindex:3 | 0.00 | 0.00 | 0 | 0.001591 | 7.719 | — | — | `operator_rate` |
| dab:stockmarket:1 | 1.00 | 1.00 | 1 | 0.0007113 | 4.741 | — | — | `operator_rate` |
| dab:stockmarket:2 | 0.00 | 0.00 | 0 | 0.001043 | 6.773 | — | — | `operator_rate` |
| dab:stockmarket:3 | 0.00 | 0.00 | 0 | 0.00133 | 7.324 | — | — | `operator_rate` |
| dab:stockmarket:4 | 0.00 | 0.00 | 0 | 0.001154 | 6.823 | — | — | `operator_rate` |
| dab:stockmarket:5 | 0.00 | 0.00 | 0 | 0.0009703 | 5.728 | — | — | `operator_rate` |
| dab:yelp:1 | 0.00 | 0.00 | 0 | 0.00093 | 5.87 | — | — | `operator_rate` |
| dab:yelp:2 | 0.00 | 0.00 | 0 | 0.001023 | 6.7 | — | — | `operator_rate` |
| dab:yelp:3 | 0.00 | 0.00 | 0 | 0.001193 | 7.091 | — | — | `operator_rate` |
| dab:yelp:4 | 0.00 | 0.00 | 0 | 0.001378 | 8.063 | — | — | `operator_rate` |
| **dab:yelp:5** ⚠️ | 0.00 | 0.00 | 0 | 0.001072 | 6.426 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |
| dab:yelp:6 | 0.00 | 0.00 | 0 | 0.001034 | 5.75 | — | — | `operator_rate` |
| dab:yelp:7 | 0.00 | 0.00 | 0 | 0.0009864 | 6.227 | — | — | `operator_rate` |

> ⚠️  Rows marked **ALL** scored 0.00 because every trial errored — the model was never successfully measured. Do not read those as quality results.

