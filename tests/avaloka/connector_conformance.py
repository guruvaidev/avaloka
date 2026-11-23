"""The connector conformance suite: the seven rules of design §B.1, as
functions any connector test can run against a live connector.

``tests/avaloka/test_connector_conformance.py`` runs it natively over the
registry (``avaloka.connectors.CONNECTORS``) with sqlite and local files;
``tests/containers/test_connector_postgres.py`` runs the same functions
against a real Postgres. A connector added to the registry without a case in
one of those places fails the registry test there.
"""

from __future__ import annotations

import copy
import json
import pickle
import socket
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

import pandas as pd
import pytest

from avaloka.connectors import (Capability, Connector, ConnectorError, Credentials, Endpoint, Query,
                                ReadReceipt, ReadResult, SAMPLE_FALLBACKS, SourceVersion)


@dataclass
class Case:
    """One connector under test and what the suite needs to know about it."""

    connector: Any
    frame: pd.DataFrame                     # the whole table the connector reads
    ref: Optional[str] = None               # the table/file reference; None = the selected one
    secret: Optional[str] = None            # a planted secret that must never surface
    mutate: Optional[Callable[[], None]] = None   # change the source, so source_version must change
    sample_methods: tuple[str, ...] = ("head",)
    extra_sinks: dict[str, Any] = field(default_factory=dict)


@contextmanager
def socket_recorder() -> Iterator[list[tuple[str, int]]]:
    """Record every TCP destination opened while the block runs."""
    seen: list[tuple[str, int]] = []
    original = socket.socket.connect

    def connect(self, address):  # type: ignore[no-untyped-def]
        if isinstance(address, tuple) and len(address) >= 2:
            seen.append((str(address[0]), int(address[1])))
        return original(self, address)

    socket.socket.connect = connect  # type: ignore[assignment]
    try:
        yield seen
    finally:
        socket.socket.connect = original  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Rule 1: declared capabilities are exactly the implemented ones
# --------------------------------------------------------------------------- #

def check_capabilities_are_declared_and_enforced(case: Case) -> None:
    c = case.connector
    assert isinstance(c, Connector), type(c)
    assert isinstance(c.capabilities, frozenset) and c.capabilities, "no capabilities declared"
    assert all(isinstance(cap, Capability) for cap in c.capabilities)
    gated = {Capability.READ: lambda: c.read(case.ref, max_rows=1, max_bytes=1 << 20),
             Capability.SAMPLE: lambda: c.sample(case.ref, n=1),
             Capability.PUSHDOWN: lambda: c.pushdown([], [], table=case.ref),
             Capability.SCHEMA: lambda: c.catalogue(),
             Capability.STREAM: lambda: c.window(case.ref or "", start=0, end=1),
             Capability.WRITE: lambda: c.write(case.ref or "", None)}
    for cap, call in gated.items():
        if cap in c.capabilities:
            call()   # declared: must work
        else:
            with pytest.raises(ConnectorError, match="capability not declared"):
                call()


# --------------------------------------------------------------------------- #
# Rule 2: bounded reads say when they stopped
# --------------------------------------------------------------------------- #

def check_reads_are_bounded(case: Case) -> None:
    c = case.connector
    n = len(case.frame)
    assert n >= 10, "the case needs at least ten rows"
    full = c.read(case.ref, max_rows=n + 10, max_bytes=1 << 30)
    assert isinstance(full, ReadResult) and isinstance(full.receipt, ReadReceipt)
    assert len(full) == n and full.truncated is False and full.receipt.stopped_by is None
    assert full.receipt.rows == n and full.receipt.bytes == full.table.nbytes > 0
    assert set(full.receipt.columns) == set(case.frame.columns)
    exact = c.read(case.ref, max_rows=n, max_bytes=1 << 30)
    assert len(exact) == n and exact.truncated is False, "a source that fits exactly is not truncated"

    rows = c.read(case.ref, max_rows=7, max_bytes=1 << 30)
    assert len(rows) == 7 and rows.truncated is True and rows.receipt.stopped_by == "max_rows"
    assert rows.receipt.rows == 7

    budget = max(1, full.table.nbytes // 3)
    small = c.read(case.ref, max_rows=n + 10, max_bytes=budget)
    assert small.truncated is True and small.receipt.stopped_by == "max_bytes"
    assert small.receipt.bytes <= budget and small.table.nbytes <= budget
    assert 0 < len(small) < n

    with pytest.raises(ConnectorError):
        c.read(case.ref, max_rows=0, max_bytes=1)


def check_samples_are_bounded_and_labelled(case: Case) -> None:
    c = case.connector
    for method in case.sample_methods:
        s = c.sample(case.ref, n=5, method=method)
        # The receipt names the method that ran: the one asked for, or its
        # declared fallback when that came back short.
        assert s.receipt.method in (method, SAMPLE_FALLBACKS.get(method))
        assert len(s) == 5 and s.receipt.rows == 5
        assert set(s.receipt.columns) == set(case.frame.columns)
    with pytest.raises(ConnectorError):
        c.sample(case.ref, n=5, method="guess")
    with pytest.raises(ConnectorError):
        c.sample(case.ref, n=0)


# --------------------------------------------------------------------------- #
# Rule 3: push-down is one parameterised, quoted SELECT that passes the allowlist
# --------------------------------------------------------------------------- #

def check_pushdown_shape(case: Case) -> None:
    c = case.connector
    if Capability.PUSHDOWN not in c.capabilities:
        with pytest.raises(ConnectorError, match="capability not declared"):
            c.pushdown([], [], table=case.ref)
        return
    # A string column with a distinctive value, so an interpolated value could
    # not hide inside a parameter name like ":p0".
    strings = [c for c in case.frame.columns if case.frame[c].dtype == object and len(str(case.frame[c].iloc[0])) > 3]
    column = str(strings[-1] if strings else case.frame.columns[0])
    value = case.frame[column].iloc[0]
    value = value.item() if hasattr(value, "item") else value
    q = c.pushdown([{"column": column, "operator": "eq", "value": value}], [column], table=case.ref)
    assert isinstance(q, Query)
    assert q.sql.lstrip().upper().startswith("SELECT ") and q.sql.count(";") == 0
    assert str(value) not in q.sql or str(value) in column, "values are bound, never interpolated"
    assert q.parameters and all(k.startswith("p") for k in q.parameters)
    got = c.read(q).to_pandas()
    expected = case.frame[case.frame[column] == value]
    assert len(got) == len(expected)
    with pytest.raises(ConnectorError):
        c.pushdown([{"column": "1; DROP TABLE x", "operator": "eq", "value": 1}], [column], table=case.ref)
    with pytest.raises(ConnectorError):
        c.read(Query(sql="DELETE FROM t"))
    with pytest.raises(ConnectorError):
        c.read(Query(sql="SELECT 1; SELECT 2"))


# --------------------------------------------------------------------------- #
# Rule 4 and 6: credentials are a Credentials object and never surface
# --------------------------------------------------------------------------- #

def _sinks(case: Case) -> dict[str, str]:
    c = case.connector
    sinks: dict[str, str] = {"repr": repr(c), "str": str(c), "credentials": repr(c.credentials)}
    if hasattr(c, "describe"):
        sinks["describe"] = c.describe()
    if Capability.SCHEMA in c.capabilities:
        sinks["catalogue"] = json.dumps(c.catalogue(), default=str)
        sinks["planner_metadata"] = json.dumps(c.planner_metadata(), default=str)
        sinks["statistics"] = json.dumps(c.statistics(case.ref or c.catalogue()["tables"][0]["name"]), default=str)
    if Capability.READ in c.capabilities:
        sinks["receipt"] = json.dumps(c.read(case.ref, max_rows=5, max_bytes=1 << 20).receipt.as_dict(), default=str)
    sinks["endpoints"] = json.dumps([e.as_dict() for e in c.endpoints()])
    sinks["source_version"] = json.dumps(c.source_version(case.ref).as_dict(), default=str)
    for name, value in case.extra_sinks.items():
        sinks[name] = json.dumps(value() if callable(value) else value, default=str)
    return sinks


def check_credentials_never_surface(case: Case) -> None:
    c = case.connector
    creds = c.credentials
    assert isinstance(creds, Credentials)
    assert "redacted" in repr(creds)
    with pytest.raises(TypeError):
        json.dumps(creds)
    with pytest.raises(TypeError):
        pickle.dumps(creds)
    with pytest.raises(TypeError):
        copy.deepcopy(creds)
    secret = case.secret or creds.reveal()
    for where, text in _sinks(case).items():
        assert secret not in text, f"the secret reached {where}"
    with pytest.raises(TypeError):
        json.dumps({"c": c})   # the connector itself is not a document either


# --------------------------------------------------------------------------- #
# Rule 5: endpoints() is complete
# --------------------------------------------------------------------------- #

def check_endpoints_are_complete(case: Case) -> None:
    c = case.connector
    declared = c.endpoints()
    assert isinstance(declared, list) and all(isinstance(e, Endpoint) for e in declared)
    allowed = {(e.host, e.port) for e in declared}
    with socket_recorder() as seen:
        if Capability.SCHEMA in c.capabilities:
            c.catalogue()
        if Capability.READ in c.capabilities:
            c.read(case.ref, max_rows=5, max_bytes=1 << 20)
        if Capability.SAMPLE in c.capabilities:
            c.sample(case.ref, n=2)
        c.source_version(case.ref)
    opened = {(host, port) for host, port in seen}
    assert opened <= allowed, f"sockets opened to {opened - allowed} which endpoints() does not list"
    if not declared:
        assert not seen, f"a connector with no endpoints opened {seen}"
    for e in declared:
        assert e.host and 0 < e.port < 65536
        assert "@" not in e.host and "/" not in e.host


# --------------------------------------------------------------------------- #
# Rule 7: source_version is deterministic for an unchanged source
# --------------------------------------------------------------------------- #

def check_source_version_is_stable_and_moves(case: Case) -> None:
    c = case.connector
    first = c.source_version(case.ref)
    again = c.source_version(case.ref)
    assert isinstance(first, SourceVersion)
    assert first == again and first.digest == again.digest
    assert json.dumps(first.as_dict(), default=str)
    if case.mutate is not None:
        case.mutate()
        moved = c.source_version(case.ref)
        assert moved != first and moved.digest != first.digest, "the source changed but the version did not"


CHECKS = (
    check_capabilities_are_declared_and_enforced,
    check_reads_are_bounded,
    check_samples_are_bounded_and_labelled,
    check_pushdown_shape,
    check_credentials_never_surface,
    check_endpoints_are_complete,
    check_source_version_is_stable_and_moves,
)
