# Product Analytics — Design (develop-1.6, 1.7-compatible)

**Status:** implemented and merged on `develop-1.6`; the export is not called by anything · **Schema:** 1.1

## What it is for

Understanding analyst experience: funnels, retention, where people get stuck,
what they retry. It records what an analyst *did* and the question they *asked*.

This is a different purpose from the Hindsight decision layer (PR #423), which
deliberately collects no question text. Neither design justifies the other.
The question is recorded inside the deployment only; it is never exported.

**Owner decision D16.5 (2026-10-31): prompts stay local; only metadata is
exported.** An earlier decision allowed the exported payload to carry the
analyst's typed question (in a name-stripped form, behind a code constant).
Typed text can contain column names, file names and data values, which breaks
the standing constraint that telemetry never transmits customer data, column
names or file names. D16.5 reversed it: the question-text export path was
removed from the code, not switched off, and every exported record is held to an
explicit metadata allowlist at the one egress point.

## Two tiers

An event exists in two representations. They are not the same thing and the
schema says so (`representations` in `event-schema.v1.json`).

| | **Stored** | **Exported** |
| --- | --- | --- |
| Where | The deployment's own database | An Avaloka-operated ingest |
| When | By default, with a working opt-out | Only when an operator sets a flag **and** an endpoint |
| Question text | `prompt`: as typed, after pattern redaction. **Column and table names included.** | **None, in any form** (D16.5). Only `prompt_chars_bucket`, one of six length bands |
| What it carries | Everything below, plus the question | **Metadata only:** event type, timings, counts and buckets, outcomes and error class names, closed-vocabulary routes, features and kinds, versions |
| Identity | `actor_id` (stable keyed pseudonym), `session_id` | A 28-day re-keyed `actor_id`, `session_id`, `install_id` |
| Is it anonymous? | **No.** | Pseudonymous behavioural data |

**The stored tier holds customer-identifying text.** A question names the
customer's columns and tables and may name people. It sits beside the data it
describes, inside the customer's deployment, which is why storing it there
discloses nothing new — but it must not be described as anonymous, and it is
not.

**The exported tier carries metadata only, and no question text** (owner
decision D16.5). The earlier measurement that led to the first removal is
recorded under "Why no question text is exported" below.

### The stored tier: retention and access

- **Retention:** 180 days (`AVALOKA_ANALYTICS_RETENTION_DAYS`), purged from the
  ingest path at most once an hour. No scheduled job: every cluster has one
  schedulable node.
- **Access:** no API route reads events back. The only readers are the export
  job and whoever holds credentials for the deployment's database. Treat
  `analytics_events.prompt` with the same access controls as the datasets
  themselves. The collector never logs a question or a rejected value.
- **Deletion:** an analyst's opt-out deletes their rows.
- **Location:** `AVALOKA_ANALYTICS_DB_URL`, else `POSTGRES_URL`, else analytics
  is off: the collector answers `enabled: false, reason: "no_store"`, stores
  nothing, and logs that once. There is no local-file default. An operator may
  still set `AVALOKA_ANALYTICS_DB_URL=sqlite:///...` explicitly; the store then
  logs that SQLite was chosen, so it cannot be mistaken for a fallback. Not
  Supabase, which is cloud-hosted and already inventoried as third-party egress.

### The exported tier: the single privacy chokepoint

`app/analytics/export.py::to_wire` is the only code that produces anything
bound for outside the cluster.

- It never reads the stored `prompt` column. A test wraps the row and fails on
  any read of it.
- It rebuilds the record field by field, re-checks every value against the
  schema, and passes the result through `allowlist.safe()`. A column that does
  not belong to the event type, or a value outside its vocabulary, is dropped
  whatever put it in the table.
- `export_pending` then holds every record to `allowlist.EXPORT_KEYS` once more
  (`allowlist.exportable`), immediately before the transport. `EXPORT_KEYS` is
  an explicit, written-out list of metadata fields, not derived from the schema:
  a field added to the schema does not leave the cluster until it is added
  there, in review.
- `tests/analytics/test_no_prompt_egress.py` captures every outbound request at
  the transport (`urllib.request.urlopen`) for a turn whose question names a
  probe column, a probe file and a free-text probe, and asserts none of them is
  in any outbound byte. A positive control shows the capture does see a prompt
  when a builder sends one.
- For every question in the test corpus, a test asserts that no word of the
  question appears anywhere in the exported record.

What an exported `question.submitted` record says about the question: that one
was asked, how (`input_method`), whether it was a retry, and which of six length
bands it fell in (`0`, `1-20`, `21-80`, `81-200`, `201-500`, `500+`). The band is
too coarse to fingerprint a question. Counts derived from the question's content
(pattern masks, name removals) are not exported.

### Why no question text is exported

An earlier draft exported the question after a **membership test**
(`names.py`): every token tested against the column, table and dataset names
registered for that analyst, and removed on a match. That test has to over-strip
to be safe — a column called `status` is also an English word, and a column name
hides inside longer tokens (`region_id`, `regions`, `byRegion`).

Measured on a synthetic set — 30 questions resembling analyst questions against
a 16-name retail schema with ordinary column names
(`tests/analytics/test_names_overstrip_rate.py`; no real analyst data exists):

- about one word in five is removed (35 of 194);
- seven questions in ten have at least one removal (21 of 30);
- about 60% of removals are plain English words that happen to be column names
  (21 of 35), 34% schema-looking references, 6% names inside longer words.

So what would arrive centrally is the question minus its most ordinary
vocabulary, with the loss concentrated where plain speech and schema vocabulary
overlap — which, for a data tool, is most of the sentence. The owner judged that
not worth an egress path guarded by an unverified lookup, and removed it.

### Question-text export was removed (D16.5)

Until 2026-10-31 a disabled question-text path remained in the code: the
collector could prepare a name-stripped copy of the question (`prompt_export`,
via `names.py`) and `export.to_wire` could send it as `prompt_stripped` if the
constant `export.EXPORT_QUESTION_TEXT` were changed. Owner decision D16.5
removed that path. There is no constant, setting or parameter that exports
question text, and the collector no longer accepts a name vocabulary or
prepares an exportable form. The `prompt_export` and `prompt_names_*` columns
remain in the table only so existing databases need no migration; nothing writes
them and nothing exports them.

`names.py` is no longer called by the product. A name-stripped question is still
typed text, so it must not be wired to anything that leaves the cluster.
Exporting any form of the question again would need a new owner decision.

## Schema

`app/analytics/schema.py` is the source of truth;
`docs/analytics/event-schema.v1.json` is generated from it and a test fails on
drift. `prompt` is the only free-text field and the only field marked
sensitive. Ids are hex or UUID only and versions are numeric, so neither can be
used as a text slot.

**Ingest rejects; export drops.** A record with an undeclared field is rejected
at ingest and reported to the UI, so a client sending something it should not is
visible. At the export boundary an unknown key is silently dropped by
`allowlist.safe()`, the same shape as `avaloka/telemetry/allowlist.py`.

**Version.** `avaloka/telemetry/version.py` holds one `SCHEMA_VERSION` for both
telemetry producers. `MAJOR.MINOR` on every batch and row. MINOR is additive
only. MAJOR is anything else; a reader supports the current and previous MAJOR,
rows are never rewritten, and `schema.UPGRADERS` holds one pure function per
MAJOR step.

### Version history, including one deliberate reset

| Version | What it is |
| --- | --- |
| 1.0 | The version module alone (PR #426). No event shape was published under it. |
| 1.1 | The first product-analytics schema. Exported representation: behavioural fields, no question text. |

**A decision, not an oversight.** Earlier commits on this branch defined 1.1
with question text in the exported representation (`prompt_stripped` and four
counters). Removing fields from a representation is not additive; by the rule
above it is a MAJOR change. It was **not** made one. 1.1 was redefined without
those fields, because:

- nothing was merged, deployed or exported under the earlier definition — no
  stored row and no received record anywhere carries it;
- a MAJOR bump exists so that readers can handle data in the old shape, and
  there is no such data;
- going to 2.0 before 1.x had ever been released would leave a permanent
  "previous MAJOR" that never existed for readers to support.

The rule binds from the first merge of a schema. After that, removing an
exported field is a MAJOR, however inconvenient. The earlier definition remains
visible in this branch's history (`abaf5bbd`).

**Identity.** The client sends none. `session_id` is minted per tab. See the
next section for `actor_id`.

## Identity: a documented control is relaxed here, and what compensates

**The control.** `avaloka/telemetry/allowlist.py` (PR #329) refuses
`mission_id`, `conversation_id` and `thread_id` because "a stable identifier
that survives a restart links a user's activity across sessions — which is what
makes an anonymous dataset re-identifiable." The Hindsight spec's weekly HMAC
epoch follows from that rule.

**The relaxation.** Product analytics stores a stable per-analyst `actor_id`.
That is a persistent identifier of exactly the refused kind. It is relaxed
because retention and retry analysis — the stated goals — cannot be computed
without one. This is not two designs differing; it is this design relaxing that
rule, in the stored tier only.

**What compensates.**

1. *Where the key comes from.* `actor_id = HMAC-SHA256(secret, user id)[:32]`.
   The secret is 32 random bytes minted by `secrets.token_hex` the first time
   the collector runs in a deployment, and stored in that deployment's
   `analytics_meta` table. It is not derived from anything, not shared between
   deployments, and there is no environment variable or setting that supplies it.
2. *Whether the key travels.* Never. It is not logged, not returned by any
   route, and not in any export payload. A test searches the wire for it.
3. *Whether exported data lets anyone recompute an id.* No. Without the secret,
   a known user id yields nothing; a test tries the install id, the session id
   and the user id as keys.
4. *Two deployments, one human.* Different secrets, so unrelated ids, stored and
   exported. Tested.
5. *Composition with `install_id`.* This was the weak point and the derivation
   was changed to fix it. As first built, the stored `actor_id` was exported
   beside `install_id`, and the pair was a durable, global, per-person key.
   **The stored `actor_id` is no longer exported.** The wire carries
   `HMAC(secret, "export:" + epoch + ":" + actor_id)`, where the epoch is a
   28-day window of the event's receipt time. The 28 days is a constant in
   `export.py`, not a setting. So:
   - inside the deployment: stable, for as long as the 180-day retention keeps rows;
   - on the wire: `(install_id, actor_id)` identifies one analyst for at most
     28 days, and cannot be linked to the stored id, to the next epoch, or to a
     user id.

**The cost, stated.** Central retention analysis is limited to 28-day windows.
Longer-horizon retention is computable only inside a deployment. If the owner
wants longer central retention, that is a decision to lengthen the constant and
should be recorded as one.

## Collector

Three routes mounted on the existing API (`app/api/server.py`):

| Route | Purpose |
| --- | --- |
| `GET /analytics/config` | `{enabled, reason, prompt_text, opted_out, schema_version}` |
| `POST /analytics/events` | A batch: ≤100 events, ≤128 KB |
| `PUT /analytics/opt-out` | `{opted_out: bool}` — the analyst's own switch |

Order on ingest: capture switch → browser signal → size → authenticate →
analyst opt-out → envelope → rate limit (600 events/min per analyst) →
per-record validation → pattern redaction → insert.

## Transport, batching, buffering, failure

Client side (for the UI work; nothing here is implemented in `ui/` yet):

- Buffer in memory; flush at 20 events or 10 s, and on `pagehide` with
  `fetch(..., {keepalive: true})`. `sendBeacon` cannot carry the `Authorization`
  header the API needs.
- Cap the buffer at 200 events; when full, drop the oldest non-error event.
- Analytics must never block or fail a user action.
- **Call through `proxyFetch`**, as `/analysis/{id}/feedback` does. The web UI's
  nginx only proxies an allowlist of path prefixes and `/analytics` is not on
  it; a direct fetch returns `index.html`.
- **Check `navigator.globalPrivacyControl` in the browser.** The collector
  honours `Sec-GPC` / `DNT`, but it is not established that the backend proxy
  forwards those headers.

| Response | Client does |
| --- | --- |
| `202` `{accepted, duplicates, rejected[]}` | Drop the batch. Rejected records are never retried. |
| `200` `{enabled: false}` | Stop sending for the session. |
| `400` / `413` | Drop the batch. |
| `401` | Drop; retry after next login. |
| `429` / `503` | Retry once after `Retry-After`, then drop. |

Delivery is at-least-once; `event_id` is unique in storage.

## Opt-out

| Switch | Who | Effect |
| --- | --- | --- |
| `AVALOKA_ANALYTICS=off` | Operator | Nothing stored, nothing exported |
| `AVALOKA_TELEMETRY=off` | Operator | Same, unless `AVALOKA_ANALYTICS=on` is set explicitly |
| `AVALOKA_ANALYTICS_PROMPT_TEXT=off` | Operator | Behavioural events kept; question text not stored locally (it is never exported either way, D16.5) |
| `PUT /analytics/opt-out` | Analyst | Their rows deleted; later events refused; never exported |
| `Sec-GPC: 1` / `DNT: 1` | Browser | Nothing stored for that request |

Capture is on by default. **Export is off by default**: it needs
`AVALOKA_ANALYTICS_EXPORT=on` *and* an `https://`
`AVALOKA_ANALYTICS_EXPORT_ENDPOINT`. There is no default host in the code.

## Pattern redaction

`app/analytics/redact.py`, applied to the stored tier before the insert. Masks
by shape: credentials, URLs and connection strings, e-mail addresses, file names
and paths, quoted literals, `d/m/y` dates, IBANs, and any run with six or more
digits. Capped at 500 characters. This is hygiene for the local copy — nobody
wants a pasted connection string sitting in an analytics table — and it means
the stored question is not strictly verbatim. It does not remove column names,
personal names or unquoted values.

Writing a redaction module or its tests? Read `REDACTION-FIXTURES.md` first.

## The 1.7 adapter seam

`export_pending()` forwards one batch past a cursor in PR #329's wire format
(gzip NDJSON), with event types prefixed `analytics.`. It is declared in the
egress inventory as `product_analytics_export`. Nothing on develop-1.6 calls it.

Before it can be enabled:

1. ~~The ingest must accept the `analytics.*` types and their fields.~~ Done.
   `telemetry_service/schema.py` accepts the twelve `analytics.*` types and, for
   each, exactly the fields `to_wire` can emit. Its set is still closed: an
   unknown type or an undeclared key rejects the batch, and the stored-tier
   fields (`prompt` and everything in `EXPORT_TEXT_KEYS`) are refused by name.
   The ingest restates this schema rather than importing it, so a change to
   `schema.EVENTS` or to what `to_wire` emits must be made on both sides;
   `tests/telemetry_ingest/test_analytics_contract.py` builds wire records with
   this package's own export and fails when they disagree. Still open under
   this item: the ingest has no report that reads an `analytics.*` event.
2. Something must call `export_pending()`. Reuse #329's uploader schedule rather
   than adding a second CronJob to a one-node cluster.
3. **Required before anything exports:** the customer-facing egress inventory /
   DPA sub-processor list must gain this channel. That document was not located
   and is not changed by this work; it is the owner's to update.

## Not done

- No UI instrumentation.
- No authenticated request has gone through the real `app/api/server.py`. It
  was imported in the CI container and the three routes answered without
  credentials; behaviour is tested through a minimal FastAPI app.
- The export has never posted to a real ingest.
- No Helm values for the switches, no SQL migration file (tables are created on
  first use), no reporting queries.
