# Scorecard — suite `track1_adaptive_sampling` (harness 0.2.0)

- Observed spend: **$3.1187** over 60 calls · ceiling $5.00 · **60 unpriced**

## Config `t1_full`

- sut: `full_data_reference` · llm: `n/a` · sampling: `full` · env: `local` · agents: `full_swarm` · trials: 5 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t1_mean_amount | 1.00 | 1.00 | 1 | 0.2 | 0.1 | 1 | — | `unknown` |
| t1_fraud_recall | 1.00 | 1.00 | 1 | 0.2 | 0.1 | 1 | — | `unknown` |

## Config `t1_random`

- sut: `random_sample` · llm: `n/a` · sampling: `random` · env: `local` · agents: `full_swarm` · trials: 5 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t1_mean_amount | 0.60 | 1.00 | 0.9665 | 0.01 | 0.005 | 0.05 | — | `unknown` |
| t1_fraud_recall | 0.00 | 0.00 | 0.4446 | 0.01 | 0.005 | 0.05 | — | `unknown` |

## Config `t1_stratified`

- sut: `stratified_sample` · llm: `n/a` · sampling: `stratified` · env: `local` · agents: `full_swarm` · trials: 5 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t1_mean_amount | 0.60 | 1.00 | 0.9666 | 0.00995 | 0.004975 | 0.04975 | — | `unknown` |
| t1_fraud_recall | 0.00 | 0.00 | 0.3452 | 0.00995 | 0.004975 | 0.04975 | — | `unknown` |

## Config `t1_reservoir`

- sut: `reservoir_sample` · llm: `n/a` · sampling: `reservoir` · env: `local` · agents: `full_swarm` · trials: 5 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t1_mean_amount | 1.00 | 1.00 | 0.9864 | 0.01 | 0.005 | 0.05 | — | `unknown` |
| t1_fraud_recall | 0.00 | 0.00 | 0.4528 | 0.01 | 0.005 | 0.05 | — | `unknown` |

## Config `t1_domain`

- sut: `domain_sample` · llm: `n/a` · sampling: `domain` · env: `local` · agents: `full_swarm` · trials: 5 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t1_mean_amount | 0.60 | 1.00 | 0.9665 | 0.01 | 0.005 | 0.05 | — | `unknown` |
| t1_fraud_recall | 1.00 | 1.00 | 0.5 | 0.01185 | 0.005925 | 0.05925 | — | `unknown` |

## Config `t1_adaptive`

- sut: `adaptive_sample` · llm: `n/a` · sampling: `adaptive` · env: `local` · agents: `full_swarm` · trials: 5 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t1_mean_amount | 1.00 | 1.00 | 0.993 | 0.1 | 0.05 | 0.5 | — | `unknown` |
| t1_fraud_recall | 0.00 | 0.00 | 0.5405 | 0.042 | 0.021 | 0.21 | — | `unknown` |

