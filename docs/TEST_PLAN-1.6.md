# Avaloka 1.6 — Master Test Plan

| | |
|---|---|
| **Workbook** | [Avaloka-1.6-Master-Test-Plan.xlsx](Avaloka-1.6-Master-Test-Plan.xlsx) |
| **Scope** | `develop-1.6` as merged, plus all 14 open PRs targeting it |
| **Size** | 126 cases · 83 P1 · 9 assignable sheets |
| **Date** | 2026-08-30 |
| **Intended use** | One sheet per team, worked independently |

## Sheets

| # | Area | Cases | Suggested owner |
|---|---|---|---|
| 0 | Coverage & Owners | — | Test lead |
| 1 | UI & Datasets | 14 | Frontend / QA |
| 2 | K8s, Cloud & Scaling | 15 | Platform / SRE |
| 3 | Agents | 23 | Agents team |
| 4 | Model Providers | 12 | Agents / Platform |
| 5 | Data Quality & Verification | 16 | Data Science / QA |
| 6 | Conversational Benchmarks | 13 | Research / QA |
| 7 | DAB, Databases & Benchmarks | 15 | Data Engineering |
| 8 | CLI, MCP & Editions | 12 | DevEx |
| 9 | Security & Privacy | 6 | Security |

Each row carries preconditions, steps, an **expected result**, the dataset or
fixture to use, and the file, doc section or PR the case was derived from — so a
disagreement about intended behaviour can be settled against the source rather
than argued. `Status` is a dropdown; `Owner`, `Actual result` and `Defect ref`
are for the tester.

## Kaggle fixtures

Download once and share a fixture bucket across the team:

| Dataset | Used for |
|---|---|
| California Housing (`housing.csv`) | Profiling, regression training, inference |
| Titanic (`train.csv`) | Classification, imputation, unknown categories |
| Instacart Market Basket | Scale, full-scan vs sample, transfers |
| IEEE-CIS Fraud | Imbalance and baseline, wide object-heavy frames, scheduling |
| YouTube Statistics | Auto Insights, ranking prompts |
| Walmart Sales | Time-series splitter selection |

## Two things to read before starting

**Known defects are included deliberately.** Cases whose expected result contains
`NOTE` or `MUST` were found during the August review cycle and are expected to
**fail today**. They exist so the team can confirm each fix rather than
rediscover the problem. There are 13, concentrated in sheets 4, 5 and 6.

**Sheet 6 has an ordering dependency.** CV-07 and CV-08 are harness defects — a
keyword-stuffed junk string currently scores 6/6, and the reply surface is graded
on an empty question. Until those are fixed, every other conversational score is
unreliable. Verify them before trusting CV-01 to CV-06.

**SE-01 is an active incident, not a test.** `.env` is tracked in git with a live
Supabase service-role key. Rotate before working sheet 9.
