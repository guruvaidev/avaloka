"""T15.1–T15.7 natively: the conformance suite over every connector in the
registry that can run without a server (sqlite, CSV, Parquet, an in-memory
and a local-directory object store, and S3 against a moto server in this
process -- the one case where the socket recorder sees a real socket), plus the
registry check that keeps a new connector from skipping it, and the
allowlist's own cases. Bites are in each check's name in
``connector_conformance.py``; the registry bite is "register a connector
outside the registry".
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import sqlalchemy

from avaloka.connectors import CONNECTORS, Capability, ConnectorError, Credentials, Query, bounded_batches
from avaloka.connectors.files import FileConnector
from avaloka.connectors.objectstore import ObjectStoreConnector
from avaloka.connectors.postgres import PostgresConnector, SQLAlchemyConnector
from avaloka.connectors.sqlguard import verify_pushdown

from tests.avaloka.connector_conformance import CHECKS, Case

#: Connectors whose conformance runs only against a server, in tests/containers.
CONTAINER_ONLY = {PostgresConnector}

SECRET = "S3cret-" + "Value-91"


def _frame(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "id": np.arange(n),
        "region": rng.choice(["north", "south", "east", "west"], n),
        "amount": rng.normal(100, 25, n).round(2),
        "note": [f"row-{i:05d}-" + "x" * 40 for i in range(n)],
    })


def _sqlite_case(tmp_path) -> Case:
    frame = _frame()
    db = tmp_path / "shop.db"
    engine = sqlalchemy.create_engine(f"sqlite:///{db}")
    frame.to_sql("orders", engine, index=False)
    engine.dispose()
    # The credential is a planted URL (a secret the sinks must never show); the
    # engine that actually answers is sqlite.
    answering = sqlalchemy.create_engine(f"sqlite:///{db}")
    c = SQLAlchemyConnector(Credentials(f"postgresql://u:{SECRET}@db.internal:5432/x"), table="orders",
                            engine=answering)

    def mutate() -> None:
        with answering.begin() as conn:
            conn.exec_driver_sql("INSERT INTO orders (id, region, amount, note) VALUES (100000, 'north', 1.0, 'n')")

    return Case(connector=c, frame=frame, ref="orders", secret=SECRET, mutate=mutate,
                sample_methods=("head", "random"))


def _file_case(tmp_path, fmt: str) -> Case:
    frame = _frame()
    path = tmp_path / f"orders.{fmt}"
    if fmt == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    c = FileConnector(path)

    def mutate() -> None:
        if fmt == "csv":
            with path.open("a") as fh:
                fh.write("100000,north,1.0,n\n")
        else:
            pd.concat([frame, frame.iloc[:1]]).to_parquet(path, index=False)

    return Case(connector=c, frame=frame, ref="orders", mutate=mutate, sample_methods=("head", "random"))


def _write_objects(fs, root: str, frame: pd.DataFrame) -> None:
    import io
    with fs.open(f"{root}/orders.csv", "wb") as fh:
        fh.write(frame.to_csv(index=False).encode())
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False)
    with fs.open(f"{root}/orders.parquet", "wb") as fh:
        fh.write(buf.getvalue())


def _memory_case(tmp_path) -> Case:
    import fsspec
    frame = _frame()
    fs = fsspec.filesystem("memory")
    root = f"/conf-{tmp_path.name}/data"
    _write_objects(fs, root, frame)
    c = ObjectStoreConnector(f"memory:/{root}")

    def mutate() -> None:
        with fs.open(f"{root}/orders.csv", "ab") as fh:
            fh.write(b"100000,north,1.0,n\n")

    return Case(connector=c, frame=frame, ref="orders.csv", mutate=mutate, sample_methods=("head", "random"))


def _local_dir_case(tmp_path) -> Case:
    import fsspec
    frame = _frame()
    root = tmp_path / "store"
    root.mkdir()
    _write_objects(fsspec.filesystem("file"), str(root), frame)
    c = ObjectStoreConnector(f"file://{root}")

    def mutate() -> None:
        with (root / "orders.csv").open("a") as fh:
            fh.write("100000,north,1.0,n\n")

    return Case(connector=c, frame=frame, ref="orders.csv", mutate=mutate, sample_methods=("head", "random"))


_MOTO = {}


def _moto_server():
    """One moto S3 server for the module; a bucket per case."""
    if "server" not in _MOTO:
        from moto.server import ThreadedMotoServer
        server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
        server.start()
        _MOTO["server"] = server
        _MOTO["port"] = server._server.socket.getsockname()[1]
    return _MOTO["port"]


def _s3_case(tmp_path) -> Case:
    import io
    import boto3
    frame = _frame()
    port = _moto_server()
    endpoint = f"http://127.0.0.1:{port}"
    bucket = f"conf-{tmp_path.name.lower()}"[:60]
    client = boto3.client("s3", endpoint_url=endpoint, region_name="us-east-1",
                          aws_access_key_id="AKIAFAKEKEY", aws_secret_access_key=SECRET)
    client.create_bucket(Bucket=bucket)
    client.put_object(Bucket=bucket, Key="raw/orders.csv", Body=frame.to_csv(index=False).encode())
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False)
    client.put_object(Bucket=bucket, Key="raw/orders.parquet", Body=buf.getvalue())
    options = {"access_key": "AKIAFAKEKEY", "secret_key": SECRET, "endpoint_url": endpoint, "region": "us-east-1"}
    c = ObjectStoreConnector(f"s3://{bucket}/raw", Credentials(json.dumps(options)))

    def mutate() -> None:
        client.put_object(Bucket=bucket, Key="raw/orders.csv",
                          Body=frame.to_csv(index=False).encode() + b"100000,north,1.0,n\n")

    return Case(connector=c, frame=frame, ref="orders.csv", secret=SECRET, mutate=mutate,
                sample_methods=("head", "random"))


NATIVE_CASES = {
    "sqlite": (SQLAlchemyConnector, _sqlite_case),
    "csv": (FileConnector, lambda tmp_path: _file_case(tmp_path, "csv")),
    "parquet": (FileConnector, lambda tmp_path: _file_case(tmp_path, "parquet")),
    "memory": (ObjectStoreConnector, _memory_case),
    "local-dir": (ObjectStoreConnector, _local_dir_case),
    "s3-moto": (ObjectStoreConnector, _s3_case),
}


def test_every_registered_connector_has_a_conformance_case():
    """T15.1's bite: a connector outside the registry, or in it without a case."""
    covered = {cls for cls, _ in NATIVE_CASES.values()} | CONTAINER_ONLY
    assert set(CONNECTORS) == covered, (set(CONNECTORS) ^ covered)
    for cls in CONNECTORS:
        assert isinstance(cls.capabilities, frozenset) and cls.capabilities, cls
    import inspect
    import avaloka.connectors as pkg
    import pkgutil
    found = set()
    for mod in pkgutil.iter_modules(pkg.__path__):
        module = __import__(f"avaloka.connectors.{mod.name}", fromlist=["*"])
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module.__name__ and isinstance(getattr(obj, "capabilities", None), frozenset) \
                    and obj.capabilities and obj.__name__ != "ConnectorBase":
                found.add(obj)
    assert found <= set(CONNECTORS), f"connector classes not registered: {found - set(CONNECTORS)}"


@pytest.mark.parametrize("check", CHECKS, ids=lambda f: f.__name__.replace("check_", ""))
@pytest.mark.parametrize("case_name", sorted(NATIVE_CASES))
def test_conformance(tmp_path, case_name, check):
    _cls, make = NATIVE_CASES[case_name]
    case = make(tmp_path)
    try:
        check(case)
    finally:
        case.connector.close()


# --------------------------------------------------------------------------- #
# The allowlist (rule 3) on its own
# --------------------------------------------------------------------------- #

GOOD = Query(sql='SELECT "id", "region" FROM "orders" WHERE "region" = :p0 AND "amount" > :p1 '
                 'AND "status" IS NULL AND "id" IN (:p2_0, :p2_1)',
             parameters={"p0": "n", "p1": 1, "p2_0": 1, "p2_1": 2}, table="orders")


def test_the_allowlist_passes_the_plan_shape_and_reports_what_it_saw():
    report = verify_pushdown(GOOD, "sqlite")
    assert report["table"] == "orders" and report["parameters"] == 4 and report["dialect"] == "sqlite"
    assert "Select" in report["nodes"] and "Literal" not in report["nodes"]
    assert verify_pushdown(GOOD, "postgresql")["dialect"] == "postgres"


@pytest.mark.parametrize("sql, why", [
    ('SELECT "id" FROM "orders"; DROP TABLE "orders"', "one statement"),
    ('DELETE FROM "orders" WHERE "id" = :p0', "SELECT"),
    ('SELECT "id" FROM "orders" WHERE "region" = \'north\'', "literal"),
    ('SELECT "id" FROM "orders" WHERE "amount" > 100', "literal"),
    ('SELECT "id" FROM "orders" WHERE "id" IN (SELECT "id" FROM "other")', "outside the plan shape"),
    ('SELECT max("amount") FROM "orders"', "outside the plan shape"),
    ('SELECT "o"."id" FROM "orders" "o" JOIN "regions" "r" ON "o"."region" = "r"."region"', "JOIN"),
    ('SELECT "id" FROM "orders" ORDER BY "id"', "ORDER"),
    ('SELECT "id" FROM "orders" LIMIT 5', "LIMIT"),
    ('SELECT * FROM "orders"', "outside the plan shape"),
    ('SELECT "id" FROM "orders" WHERE "region" = :p0', "without values"),
    ('WITH t AS (SELECT 1) SELECT "id" FROM "orders"', "WITH"),
    ('SELECT "id" FROM "orders" WHERE "region" LIKE :p0', "outside the plan shape"),
])
def test_the_allowlist_refuses_what_a_plan_cannot_produce(sql, why):
    with pytest.raises(ConnectorError, match=why):
        verify_pushdown(Query(sql=sql, parameters={} if "without" in why else {"p0": 1}), "sqlite")


def test_the_allowlist_needs_a_known_dialect():
    with pytest.raises(ConnectorError, match="dialect"):
        verify_pushdown(GOOD, "cobol")


# --------------------------------------------------------------------------- #
# bounded_batches on its own
# --------------------------------------------------------------------------- #

def test_bounded_batches_trims_to_both_bounds():
    import pyarrow as pa
    batches = [pa.RecordBatch.from_pydict({"x": list(range(i * 10, i * 10 + 10))}) for i in range(5)]
    kept, rows, used, stopped = bounded_batches(iter(batches), max_rows=25, max_bytes=1 << 20)
    assert rows == 25 and stopped == "max_rows" and sum(b.num_rows for b in kept) == 25
    kept, rows, used, stopped = bounded_batches(iter(batches), max_rows=50, max_bytes=1 << 20)
    assert rows == 50 and stopped is None, "fits exactly: not truncated"
    one = batches[0].nbytes
    kept, rows, used, stopped = bounded_batches(iter(batches), max_rows=50, max_bytes=one * 2 + 1)
    assert stopped == "max_bytes" and used <= one * 2 + 1 and 0 < rows < 50
    kept, rows, used, stopped = bounded_batches(iter([]), max_rows=5, max_bytes=5)
    assert (kept, rows, used, stopped) == ([], 0, 0, None)
    with pytest.raises(ConnectorError):
        bounded_batches(iter(batches), max_rows=0, max_bytes=1)


def test_a_connector_cannot_call_past_its_declaration():
    """T15.2's bite: make the base class a no-op."""

    class Half(FileConnector):
        capabilities = frozenset({Capability.SCHEMA})

    import pandas as pd
    import tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp())
    pd.DataFrame({"a": [1, 2]}).to_csv(d / "t.csv", index=False)
    half = Half(d / "t.csv")
    assert half.catalogue()["tables"][0]["name"] == "t"
    with pytest.raises(ConnectorError, match="capability not declared: read"):
        half.read()
    with pytest.raises(ConnectorError, match="capability not declared: sample"):
        half.sample(n=1)
    with pytest.raises(ConnectorError, match="capability not declared: stream"):
        half.window("t", start=0, end=1)


def test_endpoints_name_the_host_and_fill_the_dialect_port(tmp_path):
    """Rule 5 on the URL alone (the socket recorder needs a server): the host
    and port the connector will open, the dialect's default when the URL has
    none, and nothing for an in-process file."""
    from avaloka.connectors import Endpoint
    db = tmp_path / "e.db"
    engine = sqlalchemy.create_engine(f"sqlite:///{db}")
    pd.DataFrame({"a": [1]}).to_sql("t", engine, index=False)
    assert SQLAlchemyConnector(f"sqlite:///{db}").endpoints() == []
    with_port = SQLAlchemyConnector(Credentials("postgresql://u:pw@db.internal:6543/x"), engine=engine)
    assert with_port.endpoints() == [Endpoint(host="db.internal", port=6543)]
    assert "pw" not in json.dumps([e.as_dict() for e in with_port.endpoints()])
    assert FileConnector(_csv(tmp_path)).endpoints() == []


def _csv(tmp_path):
    path = tmp_path / "small.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(path, index=False)
    return path
