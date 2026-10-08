"""Permit-lists for the two representations of an analytics event.

An event exists in two forms and they are not the same thing:

* **stored** -- in this deployment's own database. It holds the question as the
  analyst typed it (after pattern redaction), column and table names included.
  That text identifies the customer's business. It is not anonymous.
* **exported** -- what may leave the cluster. Behavioural fields only. **No
  question text, in any form.** The one prompt-derived value is
  ``prompt_chars_bucket``, a coarse length band.

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

#: Everything the wire may carry. Built from the schema minus the sensitive
#: fields, so a field marked sensitive tomorrow leaves this set without anyone
#: remembering to remove it.
EXPORT_KEYS: FrozenSet[str] = (
    (schema.CLIENT_FIELDS - SENSITIVE_STORED)
    # prompt_chars_bucket is one of six coarse length bands ("21-80"), not a
    # length: behavioural, and far too coarse to fingerprint a question.
    | frozenset({"actor_id", "schema_version", "session_id", "ui_version", "install_id",
                 "prompt_chars_bucket"})
)

#: Keys that exist only for the disabled question-text path. They are NOT part
#: of the exported representation. ``export.to_wire`` admits them only when
#: ``export.EXPORT_QUESTION_TEXT`` is true, which it is not, and which no
#: setting can change.
EXPORT_TEXT_KEYS: FrozenSet[str] = frozenset({
    "prompt_stripped", "prompt_redactions", "prompt_names_exact",
    "prompt_names_common_word", "prompt_names_substring",
})

assert not (EXPORT_KEYS & SENSITIVE_STORED), "a sensitive stored field is on the export permit-list"


def safe(fields: Dict[str, Any], allowed: FrozenSet[str] = EXPORT_KEYS) -> Dict[str, Any]:
    """Return only the entries whose key is permitted and whose value is not ``None``."""
    if not isinstance(fields, dict):
        return {}
    return {k: v for k, v in fields.items() if k in allowed and v is not None}
