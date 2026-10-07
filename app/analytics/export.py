"""The 1.7 adapter seam: forward stored events to an Avaloka-operated ingest.

This is the ONLY module in product analytics that sends anything out of the
cluster, and on develop-1.6 nothing invokes it: no scheduler calls
``export_pending`` and no default endpoint exists. It is here so that the
outbound route is declared in the egress inventory before it is switched on,
not discovered afterwards.

What leaves when it is switched on -- stated in full, because the egress
inventory quotes this:

* the behavioural fields of every event: types, timings, outcomes, error class
  names, route templates, chart types, feature names;
* ``prompt_chars_bucket``: which of six coarse length bands a question fell in;
* ``actor_id`` -- NOT the stored one: a pseudonym re-keyed every 28 days with a
  secret that never leaves the deployment -- and ``session_id``;
* an ``install_id`` minted for this purpose.

**No question text leaves, in any form.** Not the stored ``prompt``, not a
redacted or name-stripped version of it, not counts derived from its content.
Also not: user ids, e-mail addresses, file names, column or table names, cell
values, IP addresses, or any key outside ``allowlist.EXPORT_KEYS``.

The wire format is PR #329's: gzip NDJSON, one record per line. Its ingest
currently rejects every event type this module produces
(``telemetry_service/schema.py`` has a closed four-value ``EVENT_TYPES``), so
the destination must accept the ``analytics.*`` types before this is enabled.
"""

from __future__ import annotations

import datetime as _dt
import gzip
import hashlib
import hmac
import json
import logging
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

from app.analytics import allowlist, config, redact, schema
from app.analytics.store import Store, default_store

logger = logging.getLogger(__name__)

#: How long one exported pseudonym lasts. A constant, not a setting: operational
#: limits are tunable, privacy ones are not.
#:
#: The stored ``actor_id`` is stable, which is what a persistent identifier is,
#: and persistent identifiers are what ``avaloka/telemetry/allowlist.py`` refuses
#: (``mission_id``, ``conversation_id``, ``thread_id``) because they make a
#: dataset re-identifiable. That is acceptable inside the deployment, beside the
#: user table. It is not acceptable on the wire, where ``install_id`` rides in
#: the same record: the pair would be a durable, global, per-person key. So the
#: exported value is re-keyed per epoch and the pair lasts 28 days at most.
EXPORT_ACTOR_EPOCH_DAYS = 28


def export_actor_id(secret: bytes, actor_id: str, received_at: Any) -> Optional[str]:
    """The pseudonym that leaves: keyed, and different every epoch.

    Derived from the stored ``actor_id`` with the install-local secret, so the
    receiver cannot link it to the stored value, to a user id, or to the same
    analyst in another epoch or another deployment. ``None`` -- no identity
    exported -- when there is no secret or no usable timestamp.
    """
    if not secret or not actor_id or not isinstance(received_at, _dt.datetime):
        return None
    if received_at.tzinfo is None:
        received_at = received_at.replace(tzinfo=_dt.timezone.utc)
    epoch = int(received_at.timestamp() // 86400) // EXPORT_ACTOR_EPOCH_DAYS
    message = f"export:{epoch}:{actor_id}".encode("utf-8")
    return hmac.new(secret, message, hashlib.sha256).hexdigest()[:32]


#: Whether a name-stripped form of the question may be exported. It may not.
#:
#: The owner decided this on a measurement: with the membership test doing its
#: job, about one word in five is removed and 60% of what is removed is plain
#: English that happens to be a column name -- so what would arrive centrally is
#: the question minus its most ordinary vocabulary. Not worth the risk surface.
#:
#: A constant, not a setting, on purpose. The path it guards rests on a name
#: lookup that has never run against a real session store. Turning it on must be
#: a code change with a review, not an environment variable. The tests flip it
#: to keep the path proven on constructed data; nothing else may.
EXPORT_QUESTION_TEXT = False


#: Prefix that keeps these records apart from PR #329's four process event types.
WIRE_PREFIX = "analytics."
_CURSOR_KEY = "export_cursor"
_INSTALL_KEY = "export_install_id"
_ID_COLUMNS = ("event_id", "session_id")
_EVENT_COLUMNS = ("event_id", "seq", "t_ms", "route", "turn_id", "outcome", "error_class", "duration_ms")
_TEXT_PROPS = ("prompt_redactions", "prompt_names_exact", "prompt_names_common_word",
               "prompt_names_substring")


def to_wire(row: Dict[str, Any], install_id: str, secret: bytes = b"") -> Dict[str, Any]:
    """Project one stored row onto the wire. THE privacy boundary of this package.

    Build a dict, pass it through the permit-list, emit only what comes back --
    the same discipline as ``avaloka/telemetry``. Three things hold here by
    construction, and ``tests/analytics/test_export_boundary.py`` proves each:

    * the stored ``prompt`` column is never read;
    * no question text is emitted in any form -- ``prompt_export`` is not read
      either unless ``EXPORT_QUESTION_TEXT`` is changed in code;
    * no key outside ``allowlist.EXPORT_KEYS`` and outside this event type's
      declared fields survives, whatever is in the row.
    """
    event_type = row["event_type"]
    required, optional = schema.EVENTS[event_type]
    declared = {**schema.COMMON_OPTIONAL, **optional, **schema.COMMON_REQUIRED, **required}

    props = json.loads(row.get("props") or "{}")
    candidate = {**props, **{k: row.get(k) for k in _EVENT_COLUMNS}}
    record: Dict[str, Any] = {}
    for name, spec in declared.items():
        # Re-validated, value by value, against the schema: a column that does
        # not belong to this event type, or holds something outside its
        # vocabulary, is dropped here whatever put it in the table.
        value = candidate.get(name)
        if value is None or spec.sensitive or name == "event_type":
            continue
        try:
            schema._check_value(name, spec, value)
        except schema.Rejected:
            continue
        record[name] = value
    if props.get("prompt_chars_bucket") in schema.PROMPT_CHAR_BUCKETS:
        record["prompt_chars_bucket"] = props["prompt_chars_bucket"]
    for name in _ID_COLUMNS:
        value = row.get(name)
        if isinstance(value, str) and schema._OPAQUE_ID.match(value):
            record[name] = value
    for name in ("schema_version", "ui_version"):
        value = row.get(name)
        if isinstance(value, str) and schema._VERSION.match(value):
            record[name] = value
    record["install_id"] = install_id
    # Never the stored actor_id: the per-epoch re-keyed form, or nothing.
    record["actor_id"] = export_actor_id(secret, row.get("actor_id"), row.get("received_at"))

    allowed = allowlist.EXPORT_KEYS
    if EXPORT_QUESTION_TEXT:
        # DISABLED PATH. Reached only when the constant above is changed in
        # code. Kept so the mechanism stays tested; see names.py.
        allowed = allowed | allowlist.EXPORT_TEXT_KEYS
        stripped = row.get("prompt_export")
        if stripped and config.prompt_text_enabled():
            cleaned, _ = redact.scrub(stripped)
            if cleaned:
                record["prompt_stripped"] = cleaned
                for name in _TEXT_PROPS:
                    value = props.get(name)
                    if isinstance(value, int) and not isinstance(value, bool):
                        record[name] = value

    record = allowlist.safe(record, allowed)
    record["event_type"] = WIRE_PREFIX + event_type
    return record


def _install_id(store: Store) -> str:
    existing = store.get_meta(_INSTALL_KEY)
    if existing:
        return existing
    minted = uuid.uuid4().hex
    store.set_meta(_INSTALL_KEY, minted)
    return store.get_meta(_INSTALL_KEY) or minted


def _post(endpoint: str, records: List[Dict[str, Any]], timeout: float = 15.0) -> int:
    payload = "\n".join(json.dumps(r, separators=(",", ":")) for r in records).encode("utf-8")
    request = urllib.request.Request(
        endpoint, data=gzip.compress(payload), method="POST",
        headers={"Content-Type": "application/x-ndjson", "Content-Encoding": "gzip"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - https enforced in config
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)


def export_pending(store: Optional[Store] = None, limit: int = 500, post=_post) -> Dict[str, Any]:
    """Forward one batch past the cursor. Returns a small report; never raises.

    The cursor advances on success and on a permanent rejection (4xx other than
    429), so one bad batch cannot block everything behind it. It stays put on a
    transient failure, and the same rows go again next time -- the destination
    deduplicates on ``event_id``.
    """
    endpoint = config.export_endpoint()
    if endpoint is None:
        return {"exported": 0, "reason": "export disabled"}
    try:
        store = store or default_store()
        if store is None:
            return {"exported": 0, "reason": "no store configured"}
        cursor = int(store.get_meta(_CURSOR_KEY) or 0)
        rows = store.fetch_after(cursor, limit)
        if not rows:
            return {"exported": 0, "reason": "nothing pending"}
        install_id = _install_id(store)
        secret = store.identity_secret()
        records = [to_wire(r, install_id, secret) for r in rows if not store.is_opted_out(r.get("actor_id"))]
        status = post(endpoint, records) if records else 204
        if 200 <= status < 300:
            store.set_meta(_CURSOR_KEY, str(rows[-1]["id"]))
            return {"exported": len(records), "status": status}
        if status == 429 or status >= 500:
            return {"exported": 0, "status": status, "reason": "transient; will retry"}
        store.set_meta(_CURSOR_KEY, str(rows[-1]["id"]))
        return {"exported": 0, "status": status, "reason": "permanently rejected; skipped"}
    except Exception as exc:
        logger.warning("[analytics] export failed: %s", type(exc).__name__)
        return {"exported": 0, "reason": type(exc).__name__}


if __name__ == "__main__":  # pragma: no cover - the entry point a CronJob would call
    print(json.dumps(export_pending()))
