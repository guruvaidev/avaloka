"""T16.10: ``avaloka catalog list|show|use|forget`` against a local SQLite
catalog, ``use`` binding ``(dataset, version)``, ``analyze my.<name>``
resolving to the recorded file, and the same commands over HTTP against a
stub of the API router. Bites: bind by name only (latest) for an analysis
reference; register nothing on analyze; forget without asking.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from avaloka import cli as avaloka_cli
from avaloka.catalog import local as catalog_local
from avaloka.catalog.store import CatalogStore, reset_for_tests

runner = CliRunner()


@pytest.fixture
def home(tmp_path, monkeypatch):
    reset_for_tests()
    monkeypatch.setenv(catalog_local.ENV_HOME, str(tmp_path / "home"))
    monkeypatch.setenv(catalog_local.ENV_USER, "cli-user")
    monkeypatch.delenv("AVALOKA_CATALOG_DB_URL", raising=False)
    monkeypatch.delenv("AVALOKA_API_URL", raising=False)
    monkeypatch.delenv("AVALOKA_API_TOKEN", raising=False)
    monkeypatch.delenv(catalog_local.ENV_ENABLED, raising=False)
    monkeypatch.setenv("COLUMNS", "240")   # rich wraps at the terminal width; the assertions read whole lines
    yield tmp_path / "home"
    reset_for_tests()


@pytest.fixture
def orders(tmp_path):
    path = tmp_path / "orders_2026.csv"
    pd.DataFrame({"order_id": range(50), "region": ["n", "s"] * 25, "amount": [1.5] * 50}).to_csv(path, index=False)
    return path


def _run(*args):
    result = runner.invoke(avaloka_cli.app, list(args), catch_exceptions=False)
    return result.exit_code, result.stdout


def test_loading_a_file_registers_it_and_a_reference_loads_the_same_file(home, orders):
    """Bite: register nothing; resolve a reference to the wrong file."""
    local_path, wl, resolved = avaloka_cli._load_or_die(str(orders))
    assert Path(local_path) == orders and resolved.kind == "file"
    assert (home / "catalog.db").is_file()
    store = catalog_local.store()
    (item,) = store.list()
    assert item["name"] == "my.orders_2026" and item["kind"] == "file" and item["version_no"] == 1
    assert item["locator"] == {"path": str(orders.resolve()), "filename": "orders_2026.csv", "format": "csv"}
    assert item["column_names"] == ["order_id", "region", "amount"]
    assert store.get(item["dataset_id"])["versions"][0]["source_version"]["marker"]["sha256"]
    # unchanged: a second load is not a second version
    avaloka_cli._load_or_die(str(orders))
    assert store.get(item["dataset_id"])["version_no"] == 1
    # the reference resolves to the recorded file
    again, _, _ = avaloka_cli._load_or_die("my.orders_2026")
    assert Path(again) == orders
    # a changed file is a new version, and @1 still names the old one
    with orders.open("a") as fh:
        fh.write("99,n,2.0\n")
    avaloka_cli._load_or_die(str(orders))
    assert store.get(item["dataset_id"])["version_no"] == 2
    assert store.resolve("my.orders_2026@1").version_no == 1


def test_a_reference_nobody_registered_dies_with_a_sentence(home):
    import typer
    with pytest.raises(typer.Exit) as exc:
        avaloka_cli._load_or_die("my.payroll")
    assert exc.value.exit_code == 1


def test_registration_is_off_when_asked_and_never_fails_the_load(home, orders, monkeypatch):
    monkeypatch.setenv(catalog_local.ENV_ENABLED, "off")
    avaloka_cli._load_or_die(str(orders))
    assert not (home / "catalog.db").exists()
    monkeypatch.delenv(catalog_local.ENV_ENABLED)
    monkeypatch.setattr(catalog_local, "store", lambda: (_ for _ in ()).throw(RuntimeError("no sqlalchemy")))
    local_path, _, _ = avaloka_cli._load_or_die(str(orders))   # the analysis still loads
    assert Path(local_path) == orders


def test_list_show_use_and_forget_on_the_local_catalog(home, orders):
    code, out = _run("catalog", "list")
    assert code == 0 and "No datasets yet" in out
    avaloka_cli._load_or_die(str(orders))
    store = catalog_local.store()
    (item,) = store.list()
    store.register("file", {"path": str(orders.resolve()), "filename": "orders_2026.csv", "format": "csv"},
                   [("order_id", "int64"), ("region", "object"), ("amount", "float64"), ("note", "object")],
                   dataset_id=item["dataset_id"], analysis_id="an:churn")
    assert store.get(item["dataset_id"])["version_no"] == 2

    code, out = _run("catalog", "list")
    assert code == 0 and "my.orders_2026" in out and item["dataset_id"] in out and "file" in out
    code, out = _run("catalog", "list", "payroll")
    assert code == 0 and "No datasets matching" in out
    code, out = _run("catalog", "list", "--json")
    assert code == 0 and json.loads(out)[0]["name"] == "my.orders_2026"

    code, out = _run("catalog", "show", "my.orders_2026")
    assert code == 0 and "Versions" in out and "an:churn" in out and "order_id:int64" in out.replace(" ", "")
    code, out = _run("catalog", "show", "my.nothing")
    assert code == 1 and "couldn't show" in out

    code, out = _run("catalog", "use", "my.orders_2026")
    assert code == 0 and "version 2" in out and "avaloka analyze my.orders_2026" in out and str(orders.resolve()) in out
    code, out = _run("catalog", "use", "analysis:an:churn", "--json")
    assert code == 0
    binding = json.loads(out)
    assert binding["version_no"] == 2 and binding["pinned_by"] == "an:churn" and binding["path"] == str(orders.resolve())
    code, out = _run("catalog", "use", "my.orders_2026@1", "--json")
    assert code == 0 and json.loads(out)["version_no"] == 1, "bite: bind by name only"

    code, out = runner.invoke(avaloka_cli.app, ["catalog", "forget", item["dataset_id"]], input="n\n").exit_code, ""
    assert code == 0 and store.get(item["dataset_id"])["tombstone"] is False, "declined: nothing forgotten"
    code, out = _run("catalog", "forget", item["dataset_id"], "--yes")
    assert code == 0 and "Forgot" in out and "2 version(s)" in out
    assert store.get(item["dataset_id"])["tombstone"] is True
    code, out = _run("catalog", "list")
    assert "No datasets yet" in out


# --------------------------------------------------------------------------- #
# --server: the same commands over HTTP, against a stub of the API router
# --------------------------------------------------------------------------- #

class _Stub(BaseHTTPRequestHandler):
    calls: list = []
    token = "tok-" + "cli-test"

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authed(self):
        return self.headers.get("Authorization") == f"Bearer {self.token}"

    def do_GET(self):
        _Stub.calls.append(("GET", self.path, self._authed()))
        if not self._authed():
            return self._send(401, {"detail": "Missing or invalid auth token"})
        if self.path.startswith("/catalog/datasets/u1"):
            return self._send(200, {"dataset_id": "u1", "kind": "file", "display_name": "orders.csv", "name": "my.orders",
                                    "summary": "orders.csv: file, 2 columns", "versions": [
                                        {"version_no": 1, "captured_at": "2026-11-10T00:00:00", "row_count": 3,
                                         "columns": [["a", "int64"], ["b", "object"]], "used_by": ["an:1"]}],
                                    "schema_change": None, "terms": [], "tombstone": False})
        if self.path.startswith("/catalog/datasets/"):
            return self._send(404, {"detail": "Dataset not found"})
        if self.path.startswith("/catalog/datasets"):
            return self._send(200, {"datasets": [{"dataset_id": "u1", "kind": "file", "name": "my.orders",
                                                  "version_no": 1, "columns": 2, "row_count": 3,
                                                  "captured_at": "2026-11-10T00:00:00",
                                                  "summary": "orders.csv: file, 2 columns"}], "count": 1})
        if self.path.startswith("/catalog/resolve"):
            return self._send(200, {"dataset_id": "u1", "version_id": "dsv:1", "version_no": 1, "kind": "file",
                                    "locator": {}, "display_name": "orders.csv", "columns": [["a", "int64"]],
                                    "pinned_by": None})
        self._send(404, {"detail": "no"})

    def do_DELETE(self):
        _Stub.calls.append(("DELETE", self.path, self._authed()))
        if not self._authed():
            return self._send(401, {"detail": "Missing or invalid auth token"})
        self._send(200, {"dataset_id": "u1", "tombstone": True, "removed": {"versions": 1, "terms": 0}, "derived_kept": []})

    def log_message(self, *a):  # quiet
        return


@pytest.fixture
def api():
    _Stub.calls = []
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_the_same_commands_reach_a_server_with_the_token_and_never_print_it(home, api, monkeypatch):
    monkeypatch.setenv("AVALOKA_API_URL", api)
    monkeypatch.setenv("AVALOKA_API_TOKEN", _Stub.token)
    code, out = _run("catalog", "list")
    assert code == 0 and "my.orders" in out and f"server {api}" in out and _Stub.token not in out
    code, out = _run("catalog", "show", "u1")
    assert code == 0 and "an:1" in out and "a:int64" in out.replace(" ", "")
    code, out = _run("catalog", "use", "my.orders")
    assert code == 0 and "version 1" in out and "read where it lives" in out
    code, out = _run("catalog", "forget", "u1", "--yes")
    assert code == 0 and "Forgot u1" in out
    assert all(authed for _, _, authed in _Stub.calls) and ("DELETE", "/catalog/datasets/u1", True) in _Stub.calls
    assert not (home / "catalog.db").exists(), "with a server, nothing local is touched"
    # the token is what the server checks: without it the server's 401 is relayed as a sentence
    monkeypatch.delenv("AVALOKA_API_TOKEN")
    code, out = _run("catalog", "list", "--server", api)
    assert code == 1 and "401" in out


def test_importing_the_cli_still_imports_neither_the_app_nor_sqlalchemy():
    import subprocess
    import sys
    code = ("import sys, avaloka.cli; print(','.join(sorted(m for m in sys.modules "
            "if m == 'app' or m.startswith('app.') or m.startswith('sqlalchemy'))))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"leaked: {out.stdout.strip()}"
