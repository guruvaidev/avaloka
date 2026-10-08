"""Storage for product-analytics events: three tables in the deployment's own database.

``analytics_events``  one row per event; ``event_id`` is unique, which is what
                      makes at-least-once delivery from the browser safe.
``analytics_optout``  actors who opted out. Holds the keyed pseudonym only.
``analytics_meta``    the identity secret and the export cursor.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import secrets
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import (Column, DateTime, Index, Integer, MetaData, String, Table, Text,
                        create_engine, delete, func, insert, select, update)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool

from app.analytics import config

logger = logging.getLogger(__name__)

metadata = MetaData()

#: Columns promoted out of ``props`` because funnels and retention filter on them.
PROMOTED = ("seq", "t_ms", "event_type", "route", "turn_id", "outcome",
            "error_class", "duration_ms", "prompt", "prompt_export")

events = Table(
    "analytics_events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("event_id", String(64), nullable=False, unique=True),
    Column("schema_version", String(9), nullable=False),
    Column("received_at", DateTime(timezone=True), nullable=False),
    Column("actor_id", String(32), nullable=True),
    Column("session_id", String(64), nullable=False),
    Column("ui_version", String(32), nullable=True),
    Column("seq", Integer, nullable=False),
    Column("t_ms", Integer, nullable=False),
    Column("event_type", String(40), nullable=False),
    Column("route", String(64), nullable=True),
    Column("turn_id", String(64), nullable=True),
    Column("outcome", String(16), nullable=True),
    Column("error_class", String(64), nullable=True),
    Column("duration_ms", Integer, nullable=True),
    # SENSITIVE: the question as typed (pattern-redacted), column and table
    # names included. Local only. export.py never reads this column.
    Column("prompt", Text, nullable=True),
    # The same question with every known column/table/dataset name removed.
    # NULL when the name vocabulary was unavailable or export was off at
    # capture -- in which case no question text is ever exported for the row.
    Column("prompt_export", Text, nullable=True),
    Column("props", Text, nullable=False, default="{}"),
    Index("ix_analytics_events_actor", "actor_id", "received_at"),
    Index("ix_analytics_events_session", "session_id", "seq"),
    Index("ix_analytics_events_type", "event_type", "received_at"),
)

optout = Table(
    "analytics_optout", metadata,
    Column("actor_id", String(32), primary_key=True),
    Column("at", DateTime(timezone=True), nullable=False),
)

meta = Table(
    "analytics_meta", metadata,
    Column("key", String(32), primary_key=True),
    Column("value", Text, nullable=False),
)


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


class NotConfigured(RuntimeError):
    """No database is configured for analytics. There is no fallback."""


class Store:
    def __init__(self, url: Optional[str] = None) -> None:
        url = url or config.database_url()
        if not url:
            raise NotConfigured(f"set {config.ENV_DB_URL} or POSTGRES_URL")
        self.url = url
        kwargs: Dict[str, Any] = {}
        if self.url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            if ":memory:" in self.url:
                kwargs["poolclass"] = StaticPool
        self.engine = create_engine(self.url, **kwargs)
        metadata.create_all(self.engine)
        self._secret_lock = threading.Lock()
        self._secret: Optional[bytes] = None

    # -- meta ---------------------------------------------------------------
    def get_meta(self, key: str) -> Optional[str]:
        with self.engine.connect() as conn:
            return conn.execute(select(meta.c.value).where(meta.c.key == key)).scalar_one_or_none()

    def set_meta(self, key: str, value: str) -> None:
        with self.engine.begin() as conn:
            changed = conn.execute(update(meta).where(meta.c.key == key).values(value=value)).rowcount
            if not changed:
                conn.execute(insert(meta).values(key=key, value=value))

    def identity_secret(self) -> bytes:
        """The per-install key ``actor_id`` is derived with. Minted once, kept here."""
        with self._secret_lock:
            if self._secret is None:
                existing = self.get_meta("id_secret")
                if existing is None:
                    try:
                        with self.engine.begin() as conn:
                            conn.execute(insert(meta).values(key="id_secret", value=secrets.token_hex(32)))
                    except IntegrityError:
                        pass  # another replica minted it first; read theirs
                    existing = self.get_meta("id_secret")
                self._secret = bytes.fromhex(existing or "")
            return self._secret

    # -- opt-out ------------------------------------------------------------
    def is_opted_out(self, actor_id: Optional[str]) -> bool:
        if not actor_id:
            return False
        with self.engine.connect() as conn:
            return conn.execute(
                select(optout.c.actor_id).where(optout.c.actor_id == actor_id)
            ).first() is not None

    def set_opt_out(self, actor_id: str, opted_out: bool) -> int:
        """Record the choice. Opting out also deletes what was already stored.

        Returns the number of event rows deleted.
        """
        with self.engine.begin() as conn:
            conn.execute(delete(optout).where(optout.c.actor_id == actor_id))
            if not opted_out:
                return 0
            conn.execute(insert(optout).values(actor_id=actor_id, at=_now()))
            return conn.execute(delete(events).where(events.c.actor_id == actor_id)).rowcount or 0

    # -- events -------------------------------------------------------------
    def insert_events(self, rows: List[Dict[str, Any]]) -> Tuple[int, int]:
        """Insert rows, skipping any ``event_id`` already stored. Returns (stored, duplicates)."""
        if not rows:
            return 0, 0
        ids = [r["event_id"] for r in rows]
        with self.engine.connect() as conn:
            seen = set(conn.execute(select(events.c.event_id).where(events.c.event_id.in_(ids))).scalars())
        fresh: List[Dict[str, Any]] = []
        for row in rows:
            if row["event_id"] not in seen:
                seen.add(row["event_id"])
                fresh.append(row)
        stored = 0
        try:
            with self.engine.begin() as conn:
                if fresh:
                    conn.execute(insert(events), fresh)
            stored = len(fresh)
        except IntegrityError:
            # A concurrent request stored one of these between the check and the
            # insert. Fall back to one row at a time so the rest still land.
            for row in fresh:
                try:
                    with self.engine.begin() as conn:
                        conn.execute(insert(events).values(**row))
                    stored += 1
                except IntegrityError:
                    pass
        return stored, len(rows) - stored

    def purge_older_than(self, days: int) -> int:
        cutoff = _now() - _dt.timedelta(days=days)
        with self.engine.begin() as conn:
            return conn.execute(delete(events).where(events.c.received_at < cutoff)).rowcount or 0

    def count(self) -> int:
        with self.engine.connect() as conn:
            return int(conn.execute(select(func.count()).select_from(events)).scalar_one())

    def fetch_after(self, cursor: int, limit: int) -> List[Dict[str, Any]]:
        with self.engine.connect() as conn:
            result = conn.execute(
                select(events).where(events.c.id > cursor).order_by(events.c.id).limit(limit)
            )
            return [dict(r._mapping) for r in result]

    def all_rows(self) -> List[Dict[str, Any]]:
        return self.fetch_after(0, 1_000_000)


def to_row(event: Dict[str, Any], *, schema_version: str, session_id: str,
           ui_version: Optional[str], actor_id: Optional[str],
           received_at: Optional[_dt.datetime] = None) -> Dict[str, Any]:
    """Split a validated event into promoted columns and a JSON ``props`` blob."""
    row: Dict[str, Any] = {name: event.get(name) for name in PROMOTED}
    props = {k: v for k, v in event.items() if k not in PROMOTED and k != "event_id"}
    row.update(
        event_id=event["event_id"], schema_version=schema_version,
        received_at=received_at or _now(), actor_id=actor_id,
        session_id=session_id, ui_version=ui_version,
        props=json.dumps(props, separators=(",", ":"), sort_keys=True),
    )
    return row


_default: Optional[Store] = None
_default_lock = threading.Lock()
_announced = False


def default_store() -> Optional[Store]:
    """The store the environment configures, or ``None`` when it configures none.

    ``None`` means analytics is off for this deployment; callers answer as they
    do for any other off switch. Which of the three cases applies -- nothing
    configured, a database, or SQLite because an operator named it -- is logged
    once per process, so nobody has to infer it from where rows turned up.
    """
    global _default, _announced
    with _default_lock:
        if _default is not None:
            return _default
        source = config.database_source()
        if source is None:
            if not _announced:
                _announced = True
                logger.warning(
                    "[analytics] disabled: no database configured. Set %s or POSTGRES_URL "
                    "to store events; nothing is captured and nothing is written to a "
                    "local file.", config.ENV_DB_URL)
            return None
        name, url = source
        _default = Store(url)
        # Dialect and variable name only: the URL carries credentials.
        if _default.engine.dialect.name == "sqlite":
            logger.warning(
                "[analytics] storing events in SQLite because %s is set to a sqlite URL. "
                "This is the operator's explicit setting, not a fallback; it does not "
                "survive a pod restart.", name)
        else:
            logger.info("[analytics] storing events in %s (from %s)",
                        _default.engine.dialect.name, name)
        _announced = True
        return _default


def iter_props(rows: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for row in rows:
        yield json.loads(row.get("props") or "{}")
