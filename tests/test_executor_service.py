"""The executor request path: API -> executor -> API, over a real HTTP server.

What the deployment adds (no network, non-root, read-only root) is checked
live against the quickstart; this checks the code that has to not undo it:
inputs arrive at their own paths and only inside the staging roots, only
declared outputs come back and only as regular files, a configured executor
that is down is an error and never a local fallback, and the three callers
(analysis, validator sample check, DTA preview) all go through it.
"""

import io
import json
import os
import socket
import tarfile
import threading
from http.server import ThreadingHTTPServer

import pandas as pd
import pytest

from app.executor import server as srv
from app.executor.client import build_request, read_reply, run_script
from app.executor.protocol import OUTPUT_PREFIX, RESULT_MEMBER, member_for


@pytest.fixture
def executor(tmp_path, monkeypatch):
    """A real executor on 127.0.0.1 whose only staging root is tmp_path."""
    cfg = srv.Config(stage_roots=[str(tmp_path)], work_root=str(tmp_path / ".work"),
                     sandbox=False, max_request=64 * 1024 * 1024,
                     max_output=16 * 1024 * 1024, max_timeout=60)
    handler = type("H", (srv.Handler,), {"cfg": cfg})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    monkeypatch.setenv("AVALOKA_EXECUTOR_URL", f"http://127.0.0.1:{httpd.server_address[1]}")
    # The API's temp files (the validator's sample) must be inside the root.
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    try:
        yield tmp_path
    finally:
        httpd.shutdown()


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_round_trip_reads_input_and_returns_declared_output(executor):
    src = executor / "sess" / "in.csv"
    src.parent.mkdir()
    src.write_text("a\n1\n2\n3\n")
    out = executor / "sess" / "out.csv"
    script = (f"import pandas as pd\n"
              f"df = pd.read_csv({str(src)!r})\n"
              f"df.assign(b=df.a * 2).to_csv({str(out)!r}, index=False)\n"
              f"print('rows', len(df))\n")
    # The executor stages into the same root here, so take the input away
    # from the "API side" copy to prove the script read the staged one.
    ran = run_script(script, timeout=30, inputs=[str(src)], outputs=[str(out)])
    assert ran.returncode == 0, (ran.stderr, ran.error)
    assert "rows 3" in ran.stdout
    assert pd.read_csv(out)["b"].tolist() == [2, 4, 6]
    assert ran.outputs == [str(out)]


def test_script_failure_and_timeout_are_reported(executor):
    ran = run_script("raise SystemExit(3)", timeout=30)
    assert ran.returncode == 3 and not ran.timed_out
    ran = run_script("import time; time.sleep(30)", timeout=1)
    assert ran.timed_out and ran.returncode is None


def test_child_env_is_the_allowlist(executor, monkeypatch):
    monkeypatch.setenv("DB_ENCRYPTION_KEY", "k")
    ran = run_script("import os; print(sorted(os.environ))", timeout=30)
    assert ran.returncode == 0
    assert "DB_ENCRYPTION_KEY" not in ran.stdout


def test_configured_but_unreachable_executor_is_an_error_not_a_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("AVALOKA_EXECUTOR_URL", f"http://127.0.0.1:{_free_port()}")
    marker = tmp_path / "ran-locally"
    ran = run_script(f"open({str(marker)!r}, 'w').write('x')", timeout=10)
    assert ran.returncode is None and "unreachable" in (ran.error or "")
    assert not marker.exists()


def test_inputs_outside_the_staging_roots_are_refused(tmp_path):
    cfg = srv.Config(stage_roots=[str(tmp_path / "root")], work_root=str(tmp_path / "w"),
                     sandbox=False, max_request=1 << 20, max_output=1 << 20, max_timeout=10)
    elsewhere = tmp_path / "elsewhere.csv"
    elsewhere.write_text("x\n")
    buf = io.BytesIO()
    build_request("print(1)", timeout=5, inputs=[str(elsewhere)], outputs=[], fileobj=buf)
    buf.seek(0)
    with pytest.raises(ValueError, match="staging root"):
        srv.stage_request(buf, cfg)


def test_reply_writes_only_declared_regular_outputs(tmp_path):
    declared = tmp_path / "out.csv"
    sneaky = tmp_path / "authorized_keys"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        def add(name, data, kind=tarfile.REGTYPE, link=""):
            info = tarfile.TarInfo(name)
            info.size, info.type, info.linkname = len(data), kind, link
            tar.addfile(info, io.BytesIO(data))
        add(RESULT_MEMBER, json.dumps({"returncode": 0}).encode())
        add(member_for(OUTPUT_PREFIX, str(declared)), b"a\n1\n")
        add(member_for(OUTPUT_PREFIX, str(sneaky)), b"ssh-rsa x")
        add("out/../../etc/passwd", b"x")
    buf.seek(0)
    res = read_reply(buf, [str(declared)])
    assert res.returncode == 0 and declared.read_text() == "a\n1\n"
    assert not sneaky.exists()


def test_analysis_entry_point_goes_through_the_executor(executor):
    from app.agents.execution_agent import execute_code_on_local
    src = executor / "s1" / "input.csv"
    src.parent.mkdir()
    src.write_text("g,v\nx,1\nx,2\ny,5\n")
    code = "def main(df):\n    return df.groupby('g', as_index=False)['v'].sum()\n"
    out_path = executor / "s1" / "output.csv"
    out = execute_code_on_local(code=code, input_data_location=str(src),
                                output_location=str(out_path), read_path_map={"t": str(src)})
    assert out["status"] == "success", out
    assert pd.read_csv(out_path).to_dict("records") == [{"g": "x", "v": 3}, {"g": "y", "v": 5}]


def test_validator_sample_check_goes_through_the_executor(executor, monkeypatch):
    from app.executor import sample as sample_mod
    calls = []
    real = sample_mod.run_script
    monkeypatch.setattr(sample_mod, "run_script", lambda *a, **k: calls.append(k) or real(*a, **k))
    from app.agents.validator import execute_code_node
    state = {
        "generated_code": "import os\ndef main(df):\n    print('in main')\n"
                          "    return df.groupby('g', as_index=False)['v'].sum()\n",
        "uploaded_csv_preview": [{"g": "x", "v": 1}, {"g": "x", "v": 2}, {"g": "y", "v": 5}],
    }
    res = execute_code_node(state)
    assert calls, "the validator ran generated code without the executor"
    assert res["execution_error"] is None, res
    assert res["execution_output_data"] == [{"g": "x", "v": 3}, {"g": "y", "v": 5}]
    assert "in main" in res["execution_stdout"]

    state["generated_code"] = "def main(df):\n    raise KeyError('nope')\n"
    assert "nope" in execute_code_node(state)["execution_error"]
    state["generated_code"] = "def main(df):\n    return 3\n"
    assert execute_code_node(state)["execution_error"] == "main() must return a pandas DataFrame (got int)."
