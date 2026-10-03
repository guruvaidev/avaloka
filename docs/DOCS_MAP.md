# Documentation map

What each document is for, and who it is written for. If you are not sure where
to start: **[README.md](../README.md)** for what Avaloka is,
**[INSTALL.md](INSTALL.md)** to get it running.

## Start here

| Document | For | What it is |
| --- | --- | --- |
| [README.md](../README.md) | everyone | What Avaloka is, the agent roster, a five-minute quickstart, and what is actually proven about it |
| [INSTALL.md](INSTALL.md) | anyone installing | Requirements, `scripts/install.sh`, editions, offline/air-gapped install, and the first-run failures that do not name their own cause |
| [USER_GUIDE.md](USER_GUIDE.md) | people using the product | Signing in, uploading data, asking questions, reading the trust panel. No Kubernetes, no code |
| [README_CLI.md](../README_CLI.md) | terminal users | The `avaloka` command — analyze, train, chat, batch, infer, deploy, coordinate, benchmark |

## Running it

| Document | For | What it is |
| --- | --- | --- |
| [deployment.md](deployment.md) | operators | Helm chart, kind quick start, images and tags, Ray/KubeRay, storage, upgrades, verification |
| [operations.md](operations.md) | operators | Day-two concerns: artefact persistence, MLflow, retries, inference-service lifecycle |
| [versions.md](versions.md) | operators, auditors | The single source of truth for pinned versions — Python, the dependency set, Kubernetes, and which container tags exist on the registry |
| [EDITIONS.md](EDITIONS.md) | evaluators, operators | What the open-source edition includes, what is withheld, and which gates are enforced in code rather than by absence |

## Understanding it

| Document | For | What it is |
| --- | --- | --- |
| [architecture.md](architecture.md) | engineers, reviewers | The deepest document here: the LangGraph agent graph node by node, the memory tiers, provider resolution, the dependency matrix, and a changelog of design decisions |
| [TECHNICAL_USER_GUIDE.md](TECHNICAL_USER_GUIDE.md) | technical users, integrators | The product explained to someone who will push on it — capabilities, the trust chapter, configuration, troubleshooting |
| [api.md](api.md) | integrators | The HTTP surface: authentication (and the exact env var it needs), the routes, and how to mint a token that verifies |
| [cli.md](cli.md) | integrators | The *planning* CLI (`python -m app.interfaces.cli.main`) and the MCP server tools — a different tool from the `avaloka` command above |
| [context-memory.md](context-memory.md) | engineers on the memory plane | Subsystem notes for the four memory layers: modules, configuration, timeouts and their history. Implementation detail, not a user guide |

## Evidence

| Document | For | What it is |
| --- | --- | --- |
| [benchmarks.md](benchmarks.md) | evaluators | The in-repo benchmark: what it measures, what it does not, and how to re-run it |
| [model-routing-cost.md](model-routing-cost.md) | operators choosing models | Per-agent model routing and what it costs. Read its caveat on the accuracy numbers |
| [testing.md](testing.md) | contributors | The test tiers, the eleven pytest markers, the live provider tier, and how to run each |
| [test-reports/](test-reports/) | reviewers | Dated measurement records — [reliability](test-reports/1.6-reliability-measurements.md) and [three-pillar coverage](test-reports/three-pillar-coverage.md). Snapshots, not current state |
| [TEST_PLAN-1.6.md](TEST_PLAN-1.6.md) | the test lead | The 1.6 master test plan — sheet assignments for a manual pass |

## Contributing

| Document | For | What it is |
| --- | --- | --- |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | contributors | Environment setup, the house style, how to run the suite, the project layout, and how to sign your work |
| [CHANGELOG.md](../CHANGELOG.md) | upgraders | What changed, including the breaking changes since 1.6.0 |
| [oss-branch-model.md](oss-branch-model.md) | maintainers | How `oss/1.6` is generated from `develop-1.6`, what is withheld, and the invariants that block a bad push |
| [SECURITY.md](../SECURITY.md) | everyone | How to report a vulnerability. **Defect status lives in the issue tracker, not in `docs/`** |

## Proposals, not product

[design/](design/) holds design documents for work that is **proposed and not
built**: [AGENT_EXPERIENCE.md](design/AGENT_EXPERIENCE.md),
[LOOP_GRAPH_AND_CATALOG.md](design/LOOP_GRAPH_AND_CATALOG.md) and
[TELEMETRY.md](design/TELEMETRY.md), each with a `.docx` rendering and a
[README](design/README.md) explaining how the two are kept in step. Several cite
paths that exist on other branches; those citations carry a branch note. Nothing
in that directory describes shipped behaviour.

## Elsewhere

- **[research/](research/)** — the paper and its figures. Maintained separately
  from this set; the PDFs here predate a citation fix noted in its own README.
- **The PDFs in this directory.** `Avaloka-AI-1.0-User-Guide.pdf` and
  `Avaloka-AI-1.0-Technical-User-Guide.pdf` are 1.0-era renderings of the two
  guides; where they disagree with the Markdown above, the Markdown is current.
  `Avaloka-Research-Paper.pdf` is a compiled copy of the paper — see
  `research/README.md` for which revision.
- **[images/](images/)** — image assets for the docs. The user-guide
  screenshots under `images/user-guide/` are **not yet captured**: the previous
  set was shot against an earlier UI, so the guides describe the interface in
  words instead.

## Retired

Removed in the 1.6 documentation refresh, and still in git history:
`OSS_LAUNCH_READINESS.md` (a launch plan for a date that has passed, naming
internal project identifiers), `TEST_PLAN-1.5.2.md` and `TEST_REPORT-1.5.2.md`
(QA material for a branch two releases gone, most of its blockers now false),
and `TASK-conversational-friendliness.md` (an internal ticket). See
[oss-branch-model.md](oss-branch-model.md) for why each went.
