# Scorecard — suite `dab` (harness 0.2.0)

- Observed spend: **$0.0000** over 0 calls · ceiling $1.00

## Config `dab_avaloka_native`

- sut: `avaloka_dab` · llm: `openrouter/anthropic/claude-haiku-5.5` · sampling: `entire_dataset` · env: `dab-local` · agents: `full_swarm` · trials: 1 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dab:agnews:1 | 0.00 | 0.00 | 0 | 0.0008379 | 13.85 | — | — | `operator_rate` |
| dab:agnews:2 | 0.00 | 0.00 | 0 | 0.001008 | 11.28 | — | — | `operator_rate` |
| dab:agnews:3 | 0.00 | 0.00 | 0 | 0.0009423 | 16.2 | — | — | `operator_rate` |
| dab:agnews:4 | 0.00 | 0.00 | 0 | 0.0009467 | 10.99 | — | — | `operator_rate` |
| dab:bookreview:1 | 0.00 | 0.00 | 0 | 0.0007367 | 7.408 | — | — | `operator_rate` |
| dab:bookreview:2 | 0.00 | 0.00 | 0 | 0.0008647 | 9.445 | — | — | `operator_rate` |
| dab:bookreview:3 | 0.00 | 0.00 | 0 | 0.0009994 | 9.427 | — | — | `operator_rate` |
| dab:crmarenapro:1 | 1.00 | 1.00 | 1 | 0.002543 | 21.63 | — | — | `operator_rate` |
| dab:crmarenapro:2 | 1.00 | 1.00 | 1 | 0.001582 | 13.52 | — | — | `operator_rate` |
| dab:crmarenapro:3 | 1.00 | 1.00 | 1 | 0.001838 | 15.57 | — | — | `operator_rate` |
| dab:crmarenapro:4 | 1.00 | 1.00 | 1 | 0.001391 | 13.28 | — | — | `operator_rate` |
| dab:crmarenapro:5 | 0.00 | 0.00 | 0 | 0.001385 | 12.85 | — | — | `operator_rate` |
| dab:crmarenapro:6 | 0.00 | 0.00 | 0 | 0.001917 | 17.57 | — | — | `operator_rate` |
| dab:crmarenapro:7 | 0.00 | 0.00 | 0 | 0.001398 | 13.23 | — | — | `operator_rate` |
| dab:crmarenapro:8 | 0.00 | 0.00 | 0 | 0.001483 | 13.15 | — | — | `operator_rate` |
| dab:crmarenapro:9 | 1.00 | 1.00 | 1 | 0.001166 | 10.49 | — | — | `operator_rate` |
| dab:crmarenapro:10 | 1.00 | 1.00 | 1 | 0.001423 | 13.38 | — | — | `operator_rate` |
| dab:crmarenapro:11 | 0.00 | 0.00 | 0 | 0.001583 | 13.68 | — | — | `operator_rate` |
| dab:crmarenapro:12 | 0.00 | 0.00 | 0 | 0.000998 | 9.298 | — | — | `operator_rate` |
| dab:crmarenapro:13 | 0.00 | 0.00 | 0 | 0.001284 | 10.85 | — | — | `operator_rate` |
| dab:DEPS_DEV_V1:1 | 0.00 | 0.00 | 0 | 0.001772 | 33.35 | — | — | `operator_rate` |
| dab:DEPS_DEV_V1:2 | 0.00 | 0.00 | 0 | 0.001979 | 31.24 | — | — | `operator_rate` |
| dab:GITHUB_REPOS:1 | 0.00 | 0.00 | 0 | 0.0009087 | 23.21 | — | — | `operator_rate` |
| dab:GITHUB_REPOS:2 | 0.00 | 0.00 | 0 | 0.001322 | 24.23 | — | — | `operator_rate` |
| dab:GITHUB_REPOS:3 | 1.00 | 1.00 | 1 | 0.001391 | 34.47 | — | — | `operator_rate` |
| dab:GITHUB_REPOS:4 | 1.00 | 1.00 | 1 | 0.0009846 | 21.23 | — | — | `operator_rate` |
| dab:googlelocal:1 | 0.00 | 0.00 | 0 | 0.000934 | 10.28 | — | — | `operator_rate` |
| dab:googlelocal:2 | 0.00 | 0.00 | 0 | 0.001216 | 12.28 | — | — | `operator_rate` |
| dab:googlelocal:3 | 0.00 | 0.00 | 0 | 0.001873 | 17.72 | — | — | `operator_rate` |
| dab:googlelocal:4 | 1.00 | 1.00 | 1 | 0.0008091 | 8.617 | — | — | `operator_rate` |
| dab:music_brainz_20k:1 | 0.00 | 0.00 | 0 | 0.001073 | 10.6 | — | — | `operator_rate` |
| dab:music_brainz_20k:2 | 1.00 | 1.00 | 1 | 0.0007229 | 7 | — | — | `operator_rate` |
| dab:music_brainz_20k:3 | 0.00 | 0.00 | 0 | 0.001068 | 9.683 | — | — | `operator_rate` |
| dab:PANCANCER_ATLAS:1 | 0.00 | 0.00 | 0 | 0.001526 | 14.93 | — | — | `operator_rate` |
| **dab:PANCANCER_ATLAS:2** ⚠️ | 0.00 | 0.00 | 0 | 0.001394 | 14.93 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |
| **dab:PANCANCER_ATLAS:3** ⚠️ | 0.00 | 0.00 | 0 | 0.002096 | 19.74 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |
| dab:PATENTS:1 | 0.00 | 0.00 | 0 | 0.001377 | 26.7 | — | — | `operator_rate` |
| dab:PATENTS:2 | 0.00 | 0.00 | 0 | 0.002235 | 54.09 | — | — | `operator_rate` |
| dab:PATENTS:3 | 0.00 | 0.00 | 0 | 0.002325 | 57.71 | — | — | `operator_rate` |
| dab:stockindex:1 | 1.00 | 1.00 | 1 | 0.002069 | 20.27 | — | — | `operator_rate` |
| **dab:stockindex:2** ⚠️ | 0.00 | 0.00 | 0 | 0.001474 | 15.82 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |
| **dab:stockindex:3** ⚠️ | 0.00 | 0.00 | 0 | 0.002322 | 20.4 | — | **1/1 ALL** (unknown_error:1) | `operator_rate` |
| dab:stockmarket:1 | 1.00 | 1.00 | 1 | 0.0007072 | 7.729 | — | — | `operator_rate` |
| dab:stockmarket:2 | 1.00 | 1.00 | 1 | 0.001057 | 12.58 | — | — | `operator_rate` |
| dab:stockmarket:3 | 0.00 | 0.00 | 0 | 0.001359 | 13.23 | — | — | `operator_rate` |
| dab:stockmarket:4 | 0.00 | 0.00 | 0 | 0.001263 | 13.21 | — | — | `operator_rate` |
| dab:stockmarket:5 | 1.00 | 1.00 | 1 | 0.001302 | 12.87 | — | — | `operator_rate` |
| dab:yelp:1 | 0.00 | 0.00 | 0 | 0.0009257 | 8.707 | — | — | `operator_rate` |
| dab:yelp:2 | 0.00 | 0.00 | 0 | 0.001296 | 11.13 | — | — | `operator_rate` |
| dab:yelp:3 | 0.00 | 0.00 | 0 | 0.0009859 | 9.3 | — | — | `operator_rate` |
| dab:yelp:4 | 0.00 | 0.00 | 0 | 0.001254 | 11.18 | — | — | `operator_rate` |
| dab:yelp:5 | 0.00 | 0.00 | 0 | 0.001342 | 11.45 | — | — | `operator_rate` |
| dab:yelp:6 | 0.00 | 0.00 | 0 | 0.0009086 | 8.47 | — | — | `operator_rate` |
| dab:yelp:7 | 0.00 | 0.00 | 0 | 0.001086 | 9.501 | — | — | `operator_rate` |

> ⚠️  Rows marked **ALL** scored 0.00 because every trial errored — the model was never successfully measured. Do not read those as quality results.

