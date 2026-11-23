# Scorecard — suite `track5_reliability` (harness 0.2.0)

- Observed spend: **$1.3700** over 40 calls · ceiling $5.00 · **40 unpriced**

## Config `t5_no_fault`

- sut: `fault_injecting` · llm: `local-fault-injection` · sampling: `fixed` · env: `local` · agents: `full_swarm` · trials: 5 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t5_mean_amount | 1.00 | 1.00 | 0.9886 | 0.04 | 0.02 | 0.2 | — | `unknown` |

## Config `t5_transient_no_recovery`

- sut: `fault_injecting` · llm: `local-fault-injection` · sampling: `fixed` · env: `local` · agents: `full_swarm` · trials: 5 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t5_mean_amount | 0.00 | 0.00 | 0 | 0.022 | 0.011 | 0.11 | — | `unknown` |

## Config `t5_transient_with_recovery`

- sut: `fault_injecting` · llm: `local-fault-injection` · sampling: `fixed` · env: `local` · agents: `full_swarm` · trials: 5 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t5_mean_amount | 1.00 | 1.00 | 0.9886 | 0.042 | 0.071 | 0.21 | — | `unknown` |

## Config `t5_worker_crash_with_recovery`

- sut: `fault_injecting` · llm: `local-fault-injection` · sampling: `fixed` · env: `local` · agents: `full_swarm` · trials: 5 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t5_mean_amount | 1.00 | 1.00 | 0.9886 | 0.042 | 0.071 | 0.21 | — | `unknown` |

## Config `t5_structural_blind_retry`

- sut: `fault_injecting` · llm: `local-fault-injection` · sampling: `fixed` · env: `local` · agents: `full_swarm` · trials: 5 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t5_mean_amount | 0.00 | 0.00 | 0 | 0.022 | 0.011 | 0.11 | — | `unknown` |

## Config `t5_structural_with_mitigation`

- sut: `fault_injecting` · llm: `local-fault-injection` · sampling: `fixed` · env: `local` · agents: `full_swarm` · trials: 5 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t5_mean_amount | 1.00 | 1.00 | 0.9886 | 0.042 | 0.071 | 0.21 | — | `unknown` |

## Config `t5_corrupt_parquet_no_recovery`

- sut: `fault_injecting` · llm: `local-fault-injection` · sampling: `fixed` · env: `local` · agents: `full_swarm` · trials: 5 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t5_mean_amount | 0.00 | 0.00 | 0 | 0.022 | 0.011 | 0.11 | — | `unknown` |

## Config `t5_corrupt_parquet_with_mitigation`

- sut: `fault_injecting` · llm: `local-fault-injection` · sampling: `fixed` · env: `local` · agents: `full_swarm` · trials: 5 · internet: False

| Task | Pass@1 | Best-of-N | Quality (mean) | Cost $ (mean) | Latency s (mean) | Data scanned (mean) | Errors | Price src |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t5_mean_amount | 1.00 | 1.00 | 0.9886 | 0.042 | 0.071 | 0.21 | — | `unknown` |

