"""The product-analytics event schema: the only shapes the collector will store.

This module is the single source of truth. ``docs/analytics/event-schema.v1.json``
is generated from it (``python -m app.analytics.schema``) and a test fails when
the two disagree, so the UI and the collector cannot drift apart silently.

Design rules, in the order they bind:

* **Permit-list, fail closed.** A field that is not declared for its event type
  makes the *record* invalid; it is rejected, not filtered. Filtering would
  quietly accept a client that had started sending something it should not.
* **One free-text field exists: ``prompt``.** It is the analyst's question, it
  is collected by the owner's decision, and it is the only value that passes
  through ``redact.scrub`` rather than a closed vocabulary. Every other string
  is an enum, an opaque id, or a version. Analysts type column and table names
  into questions, so ``prompt`` carries them; nothing here pretends otherwise.
* **No customer identifier.** The client never sends an identity. See
  ``identity.py`` for how ``actor_id`` is derived on the server.
* **Reuse before invention.** ``event_id``, ``event_type``, ``session_id``,
  ``seq``, ``outcome``, ``error_class``, ``duration_ms``, ``retry_count``,
  ``file_format`` and the two bucket fields carry the names, types and meanings
  they have in ``avaloka/telemetry`` (PR #329), so the 1.7 adapter is a
  projection rather than a translation.

Versioning
----------
``schema_version`` is ``"MAJOR.MINOR"`` and is carried on every batch and stored
on every row.

* MINOR is additive only: a new event type, a new optional field, a new enum
  value. A collector accepts any batch whose MAJOR it supports. A client on a
  newer MINOR than the collector gets its unknown records rejected one by one
  (and told so in the response); the rest of the batch is stored.
* MAJOR is anything else: a rename, a removal, a changed meaning, a field
  becoming required. A collector supports the current MAJOR and the previous
  one. Old rows are never rewritten; ``UPGRADERS`` holds one pure function per
  MAJOR step and readers apply them on the way out.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

# One version for both telemetry producers; see avaloka/telemetry/version.py.
from avaloka.telemetry.version import (SCHEMA_MAJOR, SCHEMA_MINOR, SCHEMA_VERSION,  # noqa: E402,F401
                                       SUPPORTED_MAJORS)

#: One pure function per MAJOR step, keyed by the version it upgrades *from*.
#: Empty at 1.x. Readers apply these; stored rows are never rewritten.
UPGRADERS: Dict[int, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}

MAX_EVENTS_PER_BATCH = 100
MAX_BATCH_BYTES = 128 * 1024
MAX_PROMPT_CHARS = 500
MAX_T_MS = 7 * 24 * 3600 * 1000      # a tab left open for a week
MAX_DURATION_MS = 24 * 3600 * 1000

#: An opaque identifier: hex digits and hyphens only, which admits a UUID in
#: either form and cannot spell a name. Deliberately a strict subset of the
#: pattern PR #329's session module accepts (``[A-Za-z0-9_-]{8,64}``), because
#: that one admits ``ravi_kumar_acme``.
_OPAQUE_ID = re.compile(r"^[0-9a-fA-F][0-9a-fA-F-]{15,63}$")
#: Numeric dotted version with an optional short suffix: "1.6.2", "1.7.0-rc1",
#: "1.6.2+3f9a1c0". Starts with a digit, so it is not a 32-character text slot.
_VERSION = re.compile(r"^\d{1,4}(?:\.\d{1,4}){0,3}(?:[-+][0-9A-Za-z.]{1,16})?$")
#: An exception or error *class name*: one identifier, never a message.
_CLASS_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,63}$")

# ---------------------------------------------------------------------------
# Closed vocabularies. Adding a value is a MINOR bump and a code review.
# ---------------------------------------------------------------------------

#: Route *templates* from ui/src/routeTree.gen.ts. A template, never a concrete
#: path: ``/shared/$token`` is a route, ``/shared/8fa2...`` is a credential.
ROUTES = frozenset({
    "/", "/analysis", "/configurations", "/dashboard", "/database", "/datasets",
    "/db-tables", "/logout", "/proanalysis", "/proanalysis/scheduled-analysis",
    "/reports", "/reports/$reportId", "/reports/generated", "/scheduled-analysis",
    "/settings", "/shared-report/$token", "/shared/$token", "/support",
    "/support-queries", "other",
})

OUTCOMES = frozenset({"success", "failed", "refused", "timeout", "cancelled"})
ENTRIES = frozenset({"direct", "login", "shared_link", "reload"})
VIEWPORTS = frozenset({"xs", "sm", "md", "lg", "xl"})
END_REASONS = frozenset({"pagehide", "logout", "idle_timeout"})
SOURCE_KINDS = frozenset({"upload", "database", "gcs", "s3", "azure", "sample", "other"})
FILE_FORMATS = frozenset({"csv", "tsv", "xlsx", "parquet", "json", "other"})
ROW_BUCKETS = frozenset({"0", "1-1e2", "1e2-1e3", "1e3-1e4", "1e4-1e5", "1e5-1e6",
                         "1e6-1e7", "1e7-1e8", "1e8+"})
COL_BUCKETS = frozenset({"0", "1-10", "10-50", "50-200", "200-1000", "1000+"})
INPUT_METHODS = frozenset({"typed", "suggestion", "rerun", "edit_resubmit"})
RESULT_KINDS = frozenset({"table", "chart", "text", "model", "file", "none"})
CHART_TYPES = frozenset({"bar", "line", "area", "scatter", "pie", "histogram", "box",
                         "heatmap", "table", "other"})
RESULT_ACTIONS = frozenset({"chart_type_changed", "code_viewed", "table_sorted",
                            "table_paged", "expanded", "copied", "exported", "shared"})
EXPORT_FORMATS = frozenset({"csv", "xlsx", "png", "pdf", "json", "html"})
FEEDBACK_TYPES = frozenset({"positive", "negative"})
AUTH_METHODS = frozenset({"password", "oauth", "sso", "magic_link"})
SURFACES = frozenset({"analysis", "proanalysis", "datasets", "database", "dashboard",
                      "reports", "scheduled_analysis", "settings", "configurations",
                      "support", "shared", "other"})
FEATURES = frozenset({
    "auto_insights", "scheduled_analysis", "report_generate", "report_view",
    "share_link_create", "email_report", "dataset_preview", "db_table_browse",
    "model_training", "code_view", "suggestion_chip", "conversation_new",
    "conversation_resume", "settings_change", "support_query",
})


@dataclass(frozen=True)
class Field:
    """One permitted field. ``kind`` selects the check in ``_check_value``."""
    kind: str                                   # enum | int | bool | id | class | text
    values: Optional[FrozenSet[str]] = None     # enum only
    lo: int = 0
    hi: int = 0
    #: Stored in the deployment's own database only. Never exported.
    sensitive: bool = False

    def describe(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"kind": self.kind}
        if self.sensitive:
            out["sensitive"] = True
            out["exported"] = False
        if self.kind == "enum":
            out["values"] = sorted(self.values or ())
        if self.kind == "int":
            out["min"], out["max"] = self.lo, self.hi
        if self.kind == "id":
            out["pattern"] = _OPAQUE_ID.pattern
        if self.kind == "class":
            out["pattern"] = _CLASS_NAME.pattern
        if self.kind == "text":
            out["max_chars"] = MAX_PROMPT_CHARS
            out["redacted_by_collector"] = True
        return out


def _enum(values: FrozenSet[str]) -> Field:
    return Field("enum", values=values)


def _int(lo: int, hi: int) -> Field:
    return Field("int", lo=lo, hi=hi)


_ID = Field("id")
_BOOL = Field("bool")
_DURATION = _int(0, MAX_DURATION_MS)

#: On every event, set by the client.
COMMON_REQUIRED: Dict[str, Field] = {
    "event_id": _ID,                      # client-minted UUID; the dedup key
    "event_type": Field("enum"),          # filled in below, once EVENTS exists
    "seq": _int(0, 10_000_000),           # per-session counter; orders events
    "t_ms": _int(0, MAX_T_MS),            # ms since session start, monotonic clock
}
COMMON_OPTIONAL: Dict[str, Field] = {
    "route": _enum(ROUTES),
}

#: event_type -> (required, optional). ``turn_id`` is client-minted per question
#: and dies with the tab; it joins a question to its outcome, its chart and its
#: feedback without ever being the server's analysis or conversation id.
EVENTS: Dict[str, Tuple[Dict[str, Field], Dict[str, Field]]] = {
    "session.started": (
        {"entry": _enum(ENTRIES)},
        {"viewport": _enum(VIEWPORTS)},
    ),
    "session.ended": (
        {},
        {"reason": _enum(END_REASONS), "active_ms": _int(0, MAX_T_MS)},
    ),
    "auth.completed": (
        {"method": _enum(AUTH_METHODS)},
        {},
    ),
    "page.viewed": (
        {"route": _enum(ROUTES)},
        {"from_route": _enum(ROUTES)},
    ),
    "dataset.connect_started": (
        {"source_kind": _enum(SOURCE_KINDS)},
        {},
    ),
    "dataset.connect_completed": (
        {"source_kind": _enum(SOURCE_KINDS), "outcome": _enum(OUTCOMES)},
        {"file_format": _enum(FILE_FORMATS), "row_count_bucket": _enum(ROW_BUCKETS),
         "column_count_bucket": _enum(COL_BUCKETS), "duration_ms": _DURATION,
         "error_class": Field("class")},
    ),
    "question.submitted": (
        {"turn_id": _ID, "input_method": _enum(INPUT_METHODS)},
        {"prompt": Field("text", sensitive=True), "turn_index": _int(0, 10_000),
         "is_retry": _BOOL, "retry_of": _ID},
    ),
    "question.completed": (
        {"turn_id": _ID, "outcome": _enum(OUTCOMES)},
        {"duration_ms": _DURATION, "first_response_ms": _DURATION,
         "error_class": Field("class"), "result_kind": _enum(RESULT_KINDS),
         "chart_type": _enum(CHART_TYPES), "retry_count": _int(0, 100)},
    ),
    "result.interacted": (
        {"turn_id": _ID, "action": _enum(RESULT_ACTIONS)},
        {"chart_type": _enum(CHART_TYPES), "export_format": _enum(EXPORT_FORMATS)},
    ),
    "feedback.submitted": (
        {"turn_id": _ID, "feedback_type": _enum(FEEDBACK_TYPES)},
        {"has_comment": _BOOL},           # never the comment itself
    ),
    "error.shown": (
        {"surface": _enum(SURFACES), "error_class": Field("class")},
        {"http_status": _int(100, 599), "turn_id": _ID},
    ),
    "feature.used": (
        {"feature": _enum(FEATURES)},
        {"surface": _enum(SURFACES)},
    ),
}

EVENT_TYPES = frozenset(EVENTS)
COMMON_REQUIRED["event_type"] = _enum(EVENT_TYPES)

#: Batch envelope. ``session_id`` is minted by the client per tab and held in
#: sessionStorage: it groups one visit and gives nothing to join two visits on.
ENVELOPE_REQUIRED: Dict[str, Field] = {"session_id": _ID}
ENVELOPE_KEYS = frozenset({"schema_version", "session_id", "ui_version", "events"})

#: Every field name a client may send, across all event types.
CLIENT_FIELDS: FrozenSet[str] = frozenset(
    set(COMMON_REQUIRED) | set(COMMON_OPTIONAL)
    | {name for req, opt in EVENTS.values() for name in (*req, *opt)}
)

#: Client fields that stay in the deployment. The export permit-list is built
#: from CLIENT_FIELDS minus this set (allowlist.py), and a test proves no
#: exported record carries one.
SENSITIVE_FIELDS: FrozenSet[str] = frozenset(
    name
    for spec_map in (COMMON_REQUIRED, COMMON_OPTIONAL, *(m for pair in EVENTS.values() for m in pair))
    for name, spec in spec_map.items() if spec.sensitive
)

#: Added by the collector, never accepted from a client. ``prompt_export`` and
#: the three ``prompt_names_*`` counters belong to the disabled question-text
#: export path (export.EXPORT_QUESTION_TEXT); they are NULL/absent in production.
SERVER_FIELDS = ("actor_id", "received_at", "schema_version", "session_id",
                 "ui_version", "prompt_chars_bucket", "prompt_redactions",
                 "prompt_export", "prompt_names_exact",
                 "prompt_names_common_word", "prompt_names_substring")

PROMPT_CHAR_BUCKETS = ("0", "1-20", "21-80", "81-200", "201-500", "500+")


def prompt_chars_bucket(n: int) -> str:
    if n <= 0:
        return "0"
    for label, hi in (("1-20", 20), ("21-80", 80), ("81-200", 200), ("201-500", 500)):
        if n <= hi:
            return label
    return "500+"


class Rejected(Exception):
    """A batch or record the collector refuses. The client must not retry it."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _check_value(name: str, spec: Field, value: Any) -> None:
    kind = spec.kind
    if kind == "bool":
        if not isinstance(value, bool):
            raise Rejected(f"{name}: not a boolean")
    elif kind == "int":
        # bool is an int in Python; a flag is not a measurement.
        if isinstance(value, bool) or not isinstance(value, int):
            raise Rejected(f"{name}: not an integer")
        if not spec.lo <= value <= spec.hi:
            raise Rejected(f"{name}: out of range")
    elif kind == "enum":
        if not isinstance(value, str) or value not in (spec.values or ()):
            # Deliberately does not echo the value: a rejected string may be
            # exactly the thing that must not be logged.
            raise Rejected(f"{name}: not in vocabulary")
    elif kind == "id":
        if not isinstance(value, str) or not _OPAQUE_ID.match(value):
            raise Rejected(f"{name}: not an opaque id")
    elif kind == "class":
        if not isinstance(value, str) or not _CLASS_NAME.match(value):
            raise Rejected(f"{name}: not a class name")
    elif kind == "text":
        # Shape only. Content is handled by redact.scrub in the collector, and
        # length by truncation there -- an over-long question is still a question.
        if not isinstance(value, str):
            raise Rejected(f"{name}: not a string")
    else:  # pragma: no cover - a schema authoring error
        raise Rejected(f"{name}: unknown field kind")


def parse_version(raw: Any) -> Tuple[int, int]:
    if not isinstance(raw, str) or not re.match(r"^\d{1,4}\.\d{1,4}$", raw):
        raise Rejected("schema_version: must be 'MAJOR.MINOR'")
    major, minor = (int(p) for p in raw.split("."))
    if major not in SUPPORTED_MAJORS:
        raise Rejected("schema_version: unsupported major")
    return major, minor


def validate_envelope(batch: Any) -> Tuple[str, str, Optional[str], List[Any]]:
    """Check the batch wrapper. Returns (schema_version, session_id, ui_version, events)."""
    if not isinstance(batch, dict):
        raise Rejected("batch: not an object")
    unknown = set(batch) - ENVELOPE_KEYS
    if unknown:
        raise Rejected("batch: unknown key")
    parse_version(batch.get("schema_version"))
    _check_value("session_id", _ID, batch.get("session_id"))
    ui_version = batch.get("ui_version")
    if ui_version is not None and (not isinstance(ui_version, str) or not _VERSION.match(ui_version)):
        raise Rejected("ui_version: not a version string")
    events = batch.get("events")
    if not isinstance(events, list):
        raise Rejected("events: not a list")
    if len(events) > MAX_EVENTS_PER_BATCH:
        raise Rejected("events: too many")
    return batch["schema_version"], batch["session_id"], ui_version, events


def validate_event(event: Any) -> Dict[str, Any]:
    """Validate one record against the permit-list. Returns a shallow copy.

    Raises ``Rejected`` for a missing required field, an unknown field, or a
    value outside its vocabulary. Never returns a partially-valid record.
    """
    if not isinstance(event, dict):
        raise Rejected("event: not an object")
    event_type = event.get("event_type")
    if not isinstance(event_type, str) or event_type not in EVENTS:
        raise Rejected("event_type: not in vocabulary")
    required, optional = EVENTS[event_type]
    permitted: Dict[str, Field] = {**COMMON_OPTIONAL, **optional, **COMMON_REQUIRED, **required}
    for name in event:
        if name not in permitted:
            raise Rejected("event: field not permitted for this event_type")
    for name in (*COMMON_REQUIRED, *required):
        if name not in event:
            raise Rejected(f"{name}: required")
    for name, value in event.items():
        _check_value(name, permitted[name], value)
    return dict(event)


def describe() -> Dict[str, Any]:
    """The schema as plain data: what the JSON artifact and the UI types are built from."""
    def fields(spec: Dict[str, Field]) -> Dict[str, Any]:
        return {k: v.describe() for k, v in sorted(spec.items())}

    from app.analytics import allowlist   # late: allowlist is built from this module

    return {
        "schema_version": SCHEMA_VERSION,
        "supported_majors": sorted(SUPPORTED_MAJORS),
        "representations": {
            "stored": {
                "where": "the deployment's own database; never leaves the cluster",
                "fields": sorted(allowlist.STORED_KEYS),
                "sensitive": sorted(SENSITIVE_FIELDS),
            },
            "exported": {
                "where": "an Avaloka-operated ingest, only when an operator enables export",
                "fields": sorted(allowlist.EXPORT_KEYS),
                "question_text": "never exported, in any form",
                "event_type_prefix": "analytics.",
            },
        },
        "limits": {
            "max_events_per_batch": MAX_EVENTS_PER_BATCH,
            "max_batch_bytes": MAX_BATCH_BYTES,
            "max_prompt_chars": MAX_PROMPT_CHARS,
        },
        "envelope": {
            "required": {"schema_version": {"kind": "version"},
                         "session_id": _ID.describe(),
                         "events": {"kind": "list"}},
            "optional": {"ui_version": {"kind": "version", "pattern": _VERSION.pattern}},
        },
        "common": {"required": fields(COMMON_REQUIRED), "optional": fields(COMMON_OPTIONAL)},
        "events": {
            name: {"required": fields(req), "optional": fields(opt)}
            for name, (req, opt) in sorted(EVENTS.items())
        },
        "server_fields": list(SERVER_FIELDS),
        "never_accepted": sorted(NEVER_ACCEPTED),
    }


#: Named temptations. Nothing depends on this list -- the permit-list above is
#: the control -- but it gives the tests something concrete to assert against
#: and tells the next reader what was considered and refused.
NEVER_ACCEPTED = frozenset({
    "user_id", "email", "username", "name", "owner_id", "profile_id", "tenant", "org",
    "analysis_id", "conversation_id", "thread_id", "mission_id", "report_id", "token",
    "file_name", "file_path", "table_name", "column_names", "columns", "schema",
    "bucket", "data_source", "connection_string", "host",
    "error_message", "traceback", "stack_trace",
    "generated_code", "output", "result", "rows", "values", "comment",
    "ip", "user_agent", "referrer", "url", "locale", "timezone",
    "row_count", "column_count",
})


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(describe(), indent=2, sort_keys=True))
