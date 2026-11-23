"""Permit-lists for the two representations of an analytics event.

An event exists in two forms and they are not the same thing:

* **stored** -- in this deployment's own database. It holds the question as the
  analyst typed it (after pattern redaction), column and table names included.
  That text identifies the customer's business. It is not anonymous.
* **exported** -- what may leave the cluster. Metadata only: event types,
  timings, counts, outcomes and error classes, closed-vocabulary feature and
  route names, versions. **No question text, in any form** (owner decision
  D16.5, 2026-10-31). The one prompt-derived value is ``prompt_chars_bucket``,
  a coarse length band -- a count, not text.

``safe()`` has the shape and the behaviour of ``avaloka/telemetry/allowlist.py``
(PR #329): build a dict, pass it through, emit only what comes back. An unknown
key is dropped. A denylist fails open; this fails closed.

Ingest and export treat an undeclared key differently, and that is deliberate,
not an oversight:

* **Ingest rejects** (``schema.validate_event``). That edge faces our own UI. A
  rejected record is reported in the response, so a client sending something
  it should not is visible. Dropping silently there would hide our own bug.
* **Export drops** (``safe`` below). At the boundary where data leaves, silence
  is the safe failure: the key simply does not go.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet

from app.analytics import schema

#: Stored fields that must never appear in an export under any name.
SENSITIVE_STORED: FrozenSet[str] = schema.SENSITIVE_FIELDS

#: Fields the collector adds to the stored form.
STORED_SERVER_KEYS: FrozenSet[str] = frozenset({
    "actor_id", "received_at", "schema_version", "session_id", "ui_version",
    "prompt_chars_bucket", "prompt_redactions", "prompt_export",
    "prompt_names_exact", "prompt_names_common_word", "prompt_names_substring",
})

STORED_KEYS: FrozenSet[str] = schema.CLIENT_FIELDS | STORED_SERVER_KEYS

#: Everything the wire may carry: an explicit allowlist of metadata fields,
#: written out rather than derived from the schema. A field added to the
#: schema tomorrow does not leave the cluster until someone adds it here, in
#: review. Every value behind these keys is an enum, a bounded number, a
#: boolean, an exception class name, an opaque random id or a version; none can
#: hold typed text, a column, table, dataset or file name, SQL, code or a value.
EXPORT_KEYS: FrozenSet[str] = frozenset({
    # what happened, and in what order
    "event_type", "event_id", "seq", "t_ms",
    # timings and counts
    "duration_ms", "first_response_ms", "active_ms", "turn_index", "retry_count",
    "http_status", "row_count_bucket", "column_count_bucket",
    # prompt_chars_bucket is one of six coarse length bands ("21-80"), not a
    # length and not text: far too coarse to fingerprint a question.
    "prompt_chars_bucket",
    # success and failure
    "outcome", "error_class", "reason",
    # closed vocabularies: where in the product, which feature, which kind
    "route", "from_route", "entry", "viewport", "method", "source_kind",
    "file_format", "input_method", "is_retry", "retry_of", "turn_id",
    "result_kind", "chart_type", "action", "export_format", "feedback_type",
    "has_comment", "surface", "feature",
    # envelope: versions and random, non-persistent ids
    "schema_version", "ui_version", "session_id", "install_id", "actor_id",
})

#: Keys that may never be on the wire under any circumstances.
_NEVER_EXPORTED: FrozenSet[str] = SENSITIVE_STORED | schema.NEVER_ACCEPTED | frozenset({
    "prompt", "prompt_export", "prompt_stripped", "prompt_redactions",
    "prompt_names_exact", "prompt_names_common_word", "prompt_names_substring",
    "received_at",
})

assert not (EXPORT_KEYS & _NEVER_EXPORTED), "a forbidden field is on the export allowlist"
assert EXPORT_KEYS <= STORED_KEYS | {"install_id"}, "an export key the stored form does not have"


def safe(fields: Dict[str, Any], allowed: FrozenSet[str] = EXPORT_KEYS) -> Dict[str, Any]:
    """Return only the entries whose key is permitted and whose value is not ``None``."""
    if not isinstance(fields, dict):
        return {}
    return {k: v for k, v in fields.items() if k in allowed and v is not None}


def exportable(record: Dict[str, Any]) -> Dict[str, Any]:
    """The egress check: what ``export_pending`` hands to the transport.

    ``safe`` against :data:`EXPORT_KEYS`, applied at the one point where records
    leave, so it holds whatever built the record.
    """
    return safe(record, EXPORT_KEYS)
