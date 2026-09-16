# Avaloka Anonymous Usage Telemetry — Design

**Status:** Proposed · **Supersedes:** *Design Doc for Stats Agent* (dual-pipeline OTEL + Redis Streams) · **Audience:** Engineering, Security, Product

---

## 1. Summary

Avaloka's open-source release needs to answer one question: **is it working for the people who installed it?** Which capabilities get used, which fail, on what shape of hardware, and where users get stuck — so support and roadmap are informed by evidence rather than by whoever files an issue.

This design proposes the smallest thing that answers it: **an append-only JSON Lines file that a small uploader batches and posts home over HTTPS.** No Redis. No OTEL collector. No tracing backend.

It is **on by default, disclosed at install, and switchable off in one click or one environment variable.** It never transmits customer data, column names, file names, or free text — only a fixed allowlist of counts, durations, versions and outcomes.

---

## 2. Review of the existing proposal

The *Design Doc for Stats Agent* is a well-constructed document, and parts of it should survive into this one. It is, however, **solving a different problem**, and that mismatch — rather than any error in its internals — is the reason to replace it.

### 2.1 What it proposes

A dual-pipeline system splitting the execution-event stream into:

- **Operational observability** — OTEL spans from `Firefly.activate()` to a local OTEL Collector sidecar over gRPC, exported to Tempo/Jaeger for lagging-agent detection.
- **Behavioural analytics** — a dedicated Redis cluster carrying an events stream (7-day retention) and a hints stream (1 hour), feeding an `AdvisoryHint` recommendation engine delivered through the openclaw conversational gateway.

A `PrivacyGateway` classifies fields into four tiers, and prompt improvements derived from telemetry must beat the DAB benchmark suite before merging.

### 2.2 Why it does not fit this use case

**It never phones home.** This is the decisive objection. Every destination in the design — Redis, Tempo, Jaeger, Postgres — is infrastructure *the customer runs*. The data lands in the customer's cluster and stays there. The requirement is that anonymous statistics reach Avaloka so we can support the release; the design contains no egress path, no endpoint, no transport, and no consent gate. It is an excellent **internal** observability design that does not address the external one.

**It requires the customer to run infrastructure to be measured.** A dedicated Redis cluster, an OTEL collector sidecar and a tracing backend are a reasonable ask of our own production deployment. They are an unreasonable ask of somebody who downloaded the open-source release to try it on a laptop. In practice the result is that telemetry works only for users who least need supporting, and returns nothing from the long tail we most want to hear from.

**Correlation identifiers are routed, not removed.** Tier 3 sends `mission_id` and `conversation_id` to Redis and withholds them from OTEL. That is a sound *internal* separation — it keeps business identifiers out of an APM tool. For data leaving a customer's premises it is the wrong control: a stable identifier that links a user's activity across sessions is exactly what makes a dataset re-identifiable, whichever pipeline carries it. Anonymous external telemetry must not contain per-mission identifiers at all.

**Two pipelines are two failure modes.** Two client libraries, two transports, two retention policies, two security reviews, and two ways for a telemetry fault to affect a user's run. The payload this use case needs is a few hundred bytes of counters. That does not justify the surface area.

### 2.3 What should be kept

Four decisions in that document are correct and are carried into this design unchanged:

| Kept | Why |
| --- | --- |
| **Allowlist, not denylist** (`ALLOWED_METADATA_KEYS`) | A denylist fails open: a field added next quarter is transmitted by default. An allowlist fails closed. This is the single most important control in either design. |
| **Emit only after the accounting record succeeds** | An unrecorded mission never emits orphaned telemetry. Ordering discipline worth preserving. |
| **Non-blocking with a bounded queue and an explicit drop policy** | Telemetry must never block or slow the critical path. If the buffer fills, events are dropped, and that is the correct trade. |
| **Benchmark-gated improvements** | No prompt or agent change is merged on telemetry evidence alone; it must strictly improve the DAB suite first. This is a genuinely good governance rule. |

---

## 3. Requirements

| # | Requirement |
| --- | --- |
| R1 | Capture usage statistics and operating context from open-source installs |
| R2 | Transmit to Avaloka for support and product decisions |
| R3 | **Never** transmit customer data, values, column names, file names, or free text |
| R4 | Nothing transmitted may be traced back to a dataset, a person, or an organisation |
| R5 | Opt-out available in the UI and at install time; **default on**, clearly disclosed |
| R6 | Must work on a laptop, in kind, and on managed Kubernetes with no added infrastructure |
| R7 | Must never block, slow, or fail a user's analysis |
| R8 | A user must be able to see exactly what would be sent, before it is sent |

R8 is not decoration. It is the mechanism by which R3 and R4 become credible rather than asserted.

---

## 4. Architecture

### 4.1 The whole system

```
  Avaloka process
        │  emit(event)            in-memory, bounded, non-blocking
        ▼
  TelemetryWriter ──────────────► /var/log/avaloka/telemetry/*.jsonl
        │                          (append-only, rotated, world-readable)
        │
        ▼
  telemetry-uploader (sidecar or cron)
        │  batch → gzip → HTTPS POST
        ▼
  https://telemetry.avaloka.ai/v1/events
```

Three components, one file format, one network call. There is no message broker, no collector, no agent to install.

### 4.2 Why log files

A file is the one dependency every deployment already has. It works identically on a laptop, in a container with an `emptyDir`, and on a node with a mounted volume. It survives a process crash — events already written are still there. It can be inspected with `cat`. If the network is unavailable for a week, the events simply wait; the uploader is stateless and picks up where it left off.

It also makes R8 trivially true: the thing we would send *is a file the user can read*.

### 4.3 Component 1 — TelemetryWriter (in-process)

```python
class TelemetryWriter:
    def emit(self, event_type: str, fields: dict) -> None:
        """Never raises, never blocks, never reaches the network."""
```

- Applies the **allowlist** before anything is written. Unknown keys are dropped, not logged, not forwarded.
- Appends one JSON object per line to the current file. Files rotate at 5 MB or daily, whichever comes first; at most 10 are retained (~50 MB ceiling).
- Writes go through a `queue.Queue(maxsize=1000)` drained by one background thread. **If the queue is full, the newest event is dropped.** A user's analysis is never delayed by bookkeeping.
- Every write is wrapped so that any exception — full disk, read-only filesystem, permissions — is swallowed after one warning. **Telemetry failing must never fail a run.**
- If telemetry is disabled, `emit()` returns immediately and no file is created.

### 4.4 Component 2 — Uploader (out-of-process)

Deliberately a separate process, because in-process network calls are how telemetry ends up in the latency path.

- Runs as a Kubernetes sidecar, a `CronJob`, or a systemd timer. On a laptop it runs opportunistically on CLI exit.
- Every 6 hours (jittered ±30 min to avoid a thundering herd on the hour): reads unsent lines, gzips, POSTs.
- Success advances a cursor file. Failure leaves the cursor untouched and retries with exponential backoff. **At-least-once delivery**; the server deduplicates on `event_id`.
- Batches are capped at 5 MB and 10,000 events.
- Honours `HTTPS_PROXY` and a custom CA bundle, because enterprise egress usually goes through one.
- If the endpoint is unreachable for longer than the retention window, the oldest files are dropped. Telemetry never grows without bound.

### 4.5 Component 3 — Ingest endpoint

- `POST /v1/events`, TLS 1.3, gzip, no authentication (an anonymous sender has no credential to present).
- **The source IP is used for rate limiting and then discarded. It is never written to storage.** An IP address is personal data in most jurisdictions and is the most obvious way an "anonymous" dataset becomes identifiable.
- Rejects payloads containing keys outside the schema, rather than accepting and filtering. Server-side allowlisting is the second half of the control.
- Retention: 400 days, then deleted.

---

## 5. What is sent

### 5.1 The allowlist

Fields not in this table are never transmitted. Adding a field requires a change to this document and a code review.

**Install context** — sent once per upload

| Field | Example | Notes |
| --- | --- | --- |
| `install_id` | `8f14e45f-ea9b-4c1d-a4a2-...` | Random UUID4 generated at install |
| `avaloka_version` | `1.6.2` | |
| `edition` | `oss` | oss / free / professional / enterprise |
| `deployment` | `kubernetes` | laptop / kind / kubernetes |
| `cloud` | `gcp` | local / gcp / aws / azure — **provider only, never project or account** |
| `os` / `arch` | `linux` / `arm64` | |
| `python_version` | `3.11` | |
| `cpu_count` / `memory_gib` | `8` / `32` | Bucketed to powers of two |

**Capability events** — one per agent activation

| Field | Example | Notes |
| --- | --- | --- |
| `event_id` | UUID4 | Server-side deduplication |
| `event_type` | `agent.completed` | Fixed vocabulary |
| `agent` | `coder` | Fixed vocabulary |
| `outcome` | `success` | success / failed / refused / timeout |
| `error_class` | `OutOfMemoryError` | **Exception class name only — never the message** |
| `duration_ms` | `1240` | |
| `retry_count` | `0` | |
| `llm_provider` / `llm_model` | `groq` / `openai/gpt-oss-120b` | |
| `tokens_in` / `tokens_out` | `1800` / `420` | |

**Dataset shape** — never dataset content

| Field | Example | Notes |
| --- | --- | --- |
| `row_count_bucket` | `1e6-1e7` | **Bucketed, never exact** |
| `column_count_bucket` | `10-50` | |
| `file_format` | `parquet` | |
| `column_type_counts` | `{"numeric": 12, "text": 3, "datetime": 1}` | Counts by type |
| `sampled` | `true` | Whether the run used a sample |

### 5.2 What is never sent, and why

| Never sent | Reason |
| --- | --- |
| Cell values, samples, aggregates | Customer data |
| **Column names** | `patient_hiv_status` is a diagnosis. A schema is data. |
| File names, table names, bucket paths | `q3_layoffs_final.csv` discloses a business event |
| Prompts, generated code, agent output | Free text always eventually contains something private |
| **Exception messages and stack traces** | The single most common accidental data leak; messages routinely embed values and paths |
| Cloud project IDs, account numbers, hostnames, usernames | Identifies the organisation |
| IP addresses | Personal data; discarded at ingest |
| `mission_id`, `conversation_id`, thread IDs | Linkable across sessions — the correction to the previous design |
| Exact row counts | `1,048,576` is close to a fingerprint for a specific file |

**The two rules worth internalising:** *free text is never safe* — anything a human or a model can write into a string will eventually contain something we promised not to collect, which is why `error_class` is transmitted and `error_message` is not. And *a schema is data* — column names describe the business and sometimes the individual.

### 5.3 The install identifier

A random UUID4, generated once at install and stored in the config directory. It is **not** derived from hostname, MAC address, or any machine property, because a derived identifier is reproducible by anyone who can observe the machine — which makes it a fingerprint rather than a pseudonym.

Its purpose is to distinguish "one install did this a thousand times" from "a thousand installs did it once". Deleting the file yields a new identity, and that is acceptable: the identifier is for counting, not for tracking.

---

## 6. Consent

**Default: on.** Disclosed prominently at install, not buried in a licence.

At install:

```
Avaloka sends anonymous usage statistics — versions, feature counts,
error types, timings. Never your data, column names, or file names.

  [Y] Yes, help improve Avaloka   (default)
  [n] No thanks

You can change this any time:  avaloka telemetry off
See exactly what is sent:      avaloka telemetry --show
```

Four ways to turn it off, all equivalent and all immediate:

| Where | How |
| --- | --- |
| Install | `./scripts/install.sh --no-telemetry` |
| Environment | `AVALOKA_TELEMETRY=off` |
| Config | `telemetry.enabled: false` |
| UI | Settings → Privacy → toggle |

`AVALOKA_TELEMETRY=off` wins over every other source, so a cluster operator can disable it fleet-wide regardless of what any individual configured.

Disabling stops collection at the source — no file is written. It does not merely suppress the upload, because a disabled feature that still writes to disk is not disabled.

**The inspection command is the point.** `avaloka telemetry --show` prints the exact JSON that would be transmitted. A privacy claim a customer can verify in five seconds is worth more than a paragraph of assurance in a policy document, and it is what a security reviewer will ask for first.

---

## 7. Comparison

| | Existing proposal | This design |
| --- | --- | --- |
| Components | OTEL collector, Redis cluster, Tempo/Jaeger, Postgres | One file, one uploader |
| Customer infrastructure | Dedicated Redis + collector sidecar + tracing backend | None |
| Reaches Avaloka | **No** | Yes, over HTTPS |
| Consent mechanism | Not specified | Default-on, four opt-outs, inspectable |
| Correlation IDs | Sent to Redis | **Not collected** |
| Works on a laptop | No | Yes |
| Survives offline | Redis-dependent | Yes; files wait |
| Failure impact on a run | Bounded queue, drop-newest | Same, plus every write exception swallowed |

---

## 8. Implementation

| Phase | Work | Depends on |
| --- | --- | --- |
| 1 | `TelemetryWriter`, allowlist, rotation, config resolution, `avaloka telemetry --show/on/off` | — |
| 2 | Uploader, cursor, backoff, proxy support; Helm sidecar and CronJob | 1 |
| 3 | Ingest endpoint, schema validation, IP-discarding rate limiter, retention job | — |
| 4 | Install prompt, UI toggle, documentation in `EDITIONS.md` and the user guide | 1 |
| 5 | Aggregation and dashboards | 3 |

Phase 1 is independently useful: with the upload disabled it becomes local diagnostics a user can attach to a support ticket by hand — which is the same data, delivered by consent, and a reasonable interim posture while phases 2 and 3 are reviewed.

### Testing

- **Allowlist enforcement** — a `WorkUnit` carrying every forbidden field emits an event containing none of them.
- **Fails closed** — a new field added to the source model does not appear in output until explicitly allowlisted.
- **Never raises** — read-only filesystem, full disk, and permission errors are each simulated; `emit()` returns normally in all three.
- **Never blocks** — 10,000 events with a stalled writer must not delay the caller measurably.
- **Consent precedence** — env beats config beats install default.
- **Disabled means no file** — not merely no upload.
- **Payload golden file** — the exact JSON is committed and diffed, so any change to what is transmitted is visible in review rather than discovered in production.

That last test is the one that keeps this design honest over time. The risk is not that the first version leaks something; it is that the eleventh version adds a field nobody reviewed.

---

## 9. Open questions

1. **Is 400-day retention right?** Long enough for year-over-year comparison; longer than some customers will accept without asking.
2. **Should Enterprise default to off?** Regulated customers may prefer an explicit opt-in, and the edition system can already express that.
3. **Do we publish the schema?** Recommended. A public allowlist is a claim a security reviewer can check, and it costs nothing.
4. **Aggregate-only alternative.** We could transmit daily counters rather than per-event records — smaller and more private, at the cost of losing sequence information for debugging. Worth deciding before phase 1 rather than after.
