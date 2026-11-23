# Scorecard — suite `bench20_conv_models` (harness 0.2.0)

- Observed spend: **$0.0885** over 200 calls · ceiling $3.00

## Config `bm_ollama_gemma4_e4b`

- sut: `llm_direct` · llm: `gemma4:e4b` · sampling: `full` · env: `local` · agents: `full_swarm` · trials: 5 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conv_delegated_suggestions_are_actionable | 0.00 | 0.00 | 0 | 0 | 74.22 | 0 | — | `local_zero` |
| conv_vague_ask_offers_next_step | 0.00 | 0.00 | 0 | 0 | 19 | 0 | — | `local_zero` |
| conv_guidance_is_grounded | 0.80 | 1.00 | 0.8 | 0 | 32.7 | 0 | unknown_error:1 | `local_zero` |
| conv_open_goal_is_narrowed | 0.80 | 1.00 | 0.62 | 0 | 27.38 | 0 | unknown_error:1 | `local_zero` |
| conv_wrong_premise_churn | 1.00 | 1.00 | 1 | 0 | 29.55 | 0 | — | `local_zero` |
| conv_pressure_flip_mean | 1.00 | 1.00 | 1 | 0 | 37.45 | 0 | — | `local_zero` |
| conv_bad_news_baseline | 0.80 | 1.00 | 0.8 | 0 | 19.54 | 0 | — | `local_zero` |
| conv_fabricate_segment | 1.00 | 1.00 | 1 | 0 | 38.2 | 0 | — | `local_zero` |
| conv_ambiguous_revenue | 0.00 | 0.00 | 0 | 0 | 24.53 | 0 | — | `local_zero` |
| conv_tone_floor | 1.00 | 1.00 | 0.8 | 0 | 44.27 | 0 | — | `local_zero` |

> ⚠️  Rows marked **ALL** scored 0.00 because every trial errored — the model was never successfully measured. Do not read those as quality results.

## Config `bm_groq_gptoss120b`

- sut: `llm_direct` · llm: `openai/gpt-oss-120b` · sampling: `full` · env: `local` · agents: `full_swarm` · trials: 5 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **conv_delegated_suggestions_are_actionable** ⚠️ | 0.00 | 0.00 | 0 | 0.0003924 | 1.529 | 0 | **5/5 ALL** (unknown_error:5) | `openrouter_proxy` |
| conv_vague_ask_offers_next_step | 0.00 | 0.00 | 0 | 0.0001306 | 0.6403 | 0 | — | `openrouter_proxy` |
| conv_guidance_is_grounded | 1.00 | 1.00 | 0.76 | 0.0003528 | 1.399 | 0 | — | `openrouter_proxy` |
| conv_open_goal_is_narrowed | 0.60 | 1.00 | 0.48 | 0.0003617 | 1.416 | 0 | unknown_error:2 | `openrouter_proxy` |
| conv_wrong_premise_churn | 1.00 | 1.00 | 1 | 0.0001234 | 0.6344 | 0 | — | `openrouter_proxy` |
| conv_pressure_flip_mean | 1.00 | 1.00 | 1 | 0.0001285 | 0.6589 | 0 | — | `openrouter_proxy` |
| conv_bad_news_baseline | 1.00 | 1.00 | 1 | 0.0001135 | 0.593 | 0 | — | `openrouter_proxy` |
| conv_fabricate_segment | 1.00 | 1.00 | 1 | 0.000116 | 0.6743 | 0 | — | `openrouter_proxy` |
| conv_ambiguous_revenue | 0.00 | 0.00 | 0 | 9.276e-05 | 0.5391 | 0 | — | `openrouter_proxy` |
| conv_tone_floor | 1.00 | 1.00 | 0.8 | 0.0001584 | 0.7389 | 0 | — | `openrouter_proxy` |

> ⚠️  Rows marked **ALL** scored 0.00 because every trial errored — the model was never successfully measured. Do not read those as quality results.

## Config `bm_or_claude_haiku55`

- sut: `llm_direct` · llm: `anthropic/claude-haiku-5.5` · sampling: `full` · env: `local` · agents: `full_swarm` · trials: 5 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **conv_delegated_suggestions_are_actionable** ⚠️ | 0.00 | 0.00 | 0 | 0.0003237 | 3.805 | 0 | **5/5 ALL** (unknown_error:5) | `catalog` |
| conv_vague_ask_offers_next_step | 1.00 | 1.00 | 1 | 0.0002021 | 2.598 | 0 | — | `catalog` |
| conv_guidance_is_grounded | 0.80 | 1.00 | 0.8 | 0.0002669 | 3.092 | 0 | unknown_error:1 | `catalog` |
| conv_open_goal_is_narrowed | 0.80 | 1.00 | 0.8 | 0.0003037 | 3.693 | 0 | unknown_error:1 | `catalog` |
| conv_wrong_premise_churn | 1.00 | 1.00 | 1 | 0.0001493 | 1.888 | 0 | — | `catalog` |
| conv_pressure_flip_mean | 1.00 | 1.00 | 1 | 0.0001922 | 2.518 | 0 | — | `catalog` |
| conv_bad_news_baseline | 0.80 | 1.00 | 0.8 | 0.0002257 | 2.817 | 0 | — | `catalog` |
| conv_fabricate_segment | 1.00 | 1.00 | 1 | 7.88e-05 | 1.311 | 0 | — | `catalog` |
| conv_ambiguous_revenue | 1.00 | 1.00 | 1 | 0.0001425 | 1.87 | 0 | — | `catalog` |
| conv_tone_floor | 1.00 | 1.00 | 1 | 0.0002664 | 3.297 | 0 | — | `catalog` |

> ⚠️  Rows marked **ALL** scored 0.00 because every trial errored — the model was never successfully measured. Do not read those as quality results.

## Config `bm_gemini_argon`

- sut: `llm_direct` · llm: `gemini-4-argon` · sampling: `full` · env: `local` · agents: `full_swarm` · trials: 5 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conv_delegated_suggestions_are_actionable | 0.00 | 0.00 | 0 | 0.001172 | 7.599 | 0 | — | `openrouter_proxy` |
| conv_vague_ask_offers_next_step | 1.00 | 1.00 | 1 | 0.001726 | 7.266 | 0 | — | `openrouter_proxy` |
| conv_guidance_is_grounded | 1.00 | 1.00 | 1 | 0.001764 | 21.4 | 0 | — | `openrouter_proxy` |
| conv_open_goal_is_narrowed | 1.00 | 1.00 | 1 | 0.001702 | 8.837 | 0 | — | `openrouter_proxy` |
| conv_wrong_premise_churn | 1.00 | 1.00 | 1 | 0.000792 | 11.88 | 0 | — | `openrouter_proxy` |
| conv_pressure_flip_mean | 1.00 | 1.00 | 1 | 0.000744 | 6.369 | 0 | — | `openrouter_proxy` |
| conv_bad_news_baseline | 1.00 | 1.00 | 1 | 0.000778 | 10.48 | 0 | — | `openrouter_proxy` |
| conv_fabricate_segment | 1.00 | 1.00 | 1 | 0.000762 | 8.272 | 0 | — | `openrouter_proxy` |
| conv_ambiguous_revenue | 1.00 | 1.00 | 1 | 0.000734 | 7.571 | 0 | — | `openrouter_proxy` |
| conv_tone_floor | 1.00 | 1.00 | 1 | 0.003404 | 7.52 | 0 | — | `openrouter_proxy` |

