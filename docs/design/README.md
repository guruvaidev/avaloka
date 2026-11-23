# Design documents

Proposals under discussion. These describe work that is **not yet built**; for
what Avaloka does today see the [Technical User Guide](../TECHNICAL_USER_GUIDE.md).

Two of these documents cite prerequisites that exist on the 1.7 branches rather
than on `develop-1.6` — the benchmark harness and the openclaw skill directory.
Each says so where it cites them; the branch notes are there because a reader
of this branch will otherwise go looking for paths that are not here.

| Document | Proposes | Status |
| --- | --- | --- |
| [TELEMETRY.md](TELEMETRY.md) | Anonymous usage statistics from open-source installs — JSONL files, batched upload, default-on with four opt-outs | Proposed. Supersedes the dual-pipeline OTEL + Redis stats-agent design; §2 explains why. |
| [AGENT_EXPERIENCE.md](AGENT_EXPERIENCE.md) | A local wiki compiling agent experience into skills, after WikiSkill (arXiv:2608.27454) | Proposed |
| [LOOP_GRAPH_AND_CATALOG.md](LOOP_GRAPH_AND_CATALOG.md) | Extending that to loop policy and graph structure; anomaly detection; a local knowledge-graph catalog | Proposed |
| [AGENT_TRACE.md](AGENT_TRACE.md) | The agent trace API, the granular planner graph, and the telemetry projection of agent steps | Built on 1.7 |
| [SWARM_DEMO.md](SWARM_DEMO.md) | Running the live swarm demo (`/swarm`) against the real API on a laptop: the exact commands, the token, what is assumed | Built on 1.7 |
| [2.0-ADVANCED-PLAN.md](2.0-ADVANCED-PLAN.md) | Evaluation of the owner's advanced 2.0 feature set against integration/2.0: what exists, the L1/L2/L3 layering, and staffing | Owner, team lead, stream owners |
| [1.7-INSTANT-ANALYSIS.md](1.7-INSTANT-ANALYSIS.md) | Instant Analysis: a versioned recipe library over DuckDB and scipy/statsmodels that emits Outcome Reports and notebooks, routing, metering, a 90-day PR plan. Internal; withheld from the public tree by `oss/manifest.yaml` | Owner, coordinator, the PR owners in its §8 |

Each is also present as a `.docx` for circulation. **The markdown is the source**
— review and amend that; the Word files are regenerated from it with `pandoc`.
The output names do not match the input names, so pass both explicitly:

```bash
pandoc TELEMETRY.md            -o Avaloka-Telemetry-Design.docx                   --toc
pandoc AGENT_EXPERIENCE.md     -o Avaloka-Agent-Experience-WikiSkill-Design.docx  --toc
pandoc LOOP_GRAPH_AND_CATALOG.md -o Avaloka-Experience-Loop-Graph-Catalog-Design.docx --toc
```

## The through-line

Three memories at three altitudes, all local:

- **Structure** — the knowledge graph: what exists and what depends on what.
- **Experience** — the wiki: what went wrong and what worked.
- **Procedure** — skills, loop policy and graph edits, compiled from the other two.

Only compiled procedure is ever a candidate for sharing, and only by explicit
contribution. Telemetry carries counters home; the wiki and the graph never
leave the customer.
