import base64
import os
from io import StringIO
import re
import pandas as pd
import pytest
import json

# Safer than "from app.agents import execution_agent" depending on __init__.py exports
import app.agents.execution_agent as exec_mod

CELERY_SSH_SERVER = json.loads(os.getenv("CELERY_SSH_SERVERS") or '["localhost"]')[0]


def _base_state(**overrides):
    """
    execution_agent_* functions treat ETLState as a dict-like object.
    A plain dict works for integration tests.
    """
    state = {
        "messages": [],
        "coder_definition": {"code": ""},
        "coder_raw_response": None,
        "generated_code": None,
        "data_source_location": None,
        "output_location": None,
        "infrastructure_provisioned": {},
    }
    state.update(overrides)
    return state


# -------------------------
# _strip_markdown_code_fences
# -------------------------

def test_strip_markdown_code_fences_strips_wrapped_block():
    code = "```python\nprint('hi')\n```"
    assert exec_mod._strip_markdown_code_fences(code).strip() == "print('hi')"


def test_strip_markdown_code_fences_noop_when_not_wrapped():
    code = "print('hi')"
    assert exec_mod._strip_markdown_code_fences(code) == code


# -------------------------
# get_service_details
# -------------------------

def test_get_service_details_fails_when_not_provisioned():
    out = exec_mod.get_service_details({"status": "nope", "details": []})
    assert out["status"] == "failed"
    assert "not properly provisioned" in out["message"].lower()


def test_get_service_details_fails_when_ip_or_port_missing():
    infra = {
        "status": "provisioned",
        "details": [
            {
                "step_name": "Get infra-agent-service IP/Port",
                "status": "SUCCESS",
                "details": {"ip": "1.2.3.4"},  # port missing
            }
        ],
    }
    out = exec_mod.get_service_details(infra)
    assert out["status"] == "failed"
    assert "ip and port not found" in out["message"].lower()


def test_get_service_details_success_extracts_ip_port():
    infra = {
        "status": "provisioned",
        "details": [
            {
                "step_name": "Get infra-agent-service IP/Port",
                "status": "SUCCESS",
                "details": {"ip": "1.2.3.4", "port": "8080"},
            }
        ],
    }
    out = exec_mod.get_service_details(infra)
    assert out["status"] == "success"
    assert out["service_ip"] == "1.2.3.4"
    assert out["service_port"] == "8080"


# -------------------------
# execute_code_on_k8s (payload + path rewrite + error handling)
# -------------------------

def test_execute_code_on_k8s_rewrites_paths_and_includes_csv_payload(tmp_path, monkeypatch):
    in_csv = tmp_path / "in.csv"
    in_csv.write_text("a,b\n1,2\n", encoding="utf-8")

    out_csv = tmp_path / "out.csv"

    # include both forward-slash and backslash variants to exercise replacement logic
    in_forward = str(in_csv).replace("\\", "/")
    out_forward = str(out_csv).replace("\\", "/")
    in_back = str(in_csv).replace("/", "\\")
    out_back = str(out_csv).replace("/", "\\")

    code = f"""
import pandas as pd
df = pd.read_csv("{in_forward}")
df.to_csv("{out_back}", index=False)
# also mention as single-quote variants
df2 = pd.read_csv('{in_back}')
df2.to_csv('{out_forward}', index=False)
"""

    captured = {}

    class _Resp:
        status_code = 200
        headers = {"x": "y"}

        def raise_for_status(self):
            return None

        def json(self):
            return {"status": "success", "output": None}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["payload"] = json
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(exec_mod.requests, "post", fake_post)

    out = exec_mod.execute_code_on_k8s(
        code=code,
        service_ip="10.0.0.1",
        service_port="9999",
        data_source_location=str(in_csv),
        output_location=str(out_csv),
        timeout=30,
    )

    assert captured["url"] == "http://10.0.0.1:9999/execute"
    assert "code" in captured["payload"]

    decoded_code = base64.b64decode(captured["payload"]["code"]).decode("utf-8")
    assert "input_file" in decoded_code
    assert "output_file" in decoded_code
    assert str(in_csv) not in decoded_code
    assert str(out_csv) not in decoded_code

    assert "data" in captured["payload"]
    decoded_csv = base64.b64decode(captured["payload"]["data"]).decode("utf-8")
    assert decoded_csv.strip() == "a,b\n1,2"

    assert "execution_time" in out


def test_execute_code_on_k8s_skips_data_payload_if_input_file_missing(tmp_path, monkeypatch):
    missing = tmp_path / "missing.csv"

    class _Resp:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {"status": "success", "output": None}

    captured = {}

    def fake_post(url, json, timeout):
        captured["payload"] = json
        return _Resp()

    monkeypatch.setattr(exec_mod.requests, "post", fake_post)

    exec_mod.execute_code_on_k8s(
        code="print('x')",
        service_ip="10.0.0.1",
        service_port="9999",
        data_source_location=str(missing),
        output_location=None,
    )

    assert "code" in captured["payload"]
    assert "data" not in captured["payload"]  # important branch


def test_execute_code_on_k8s_returns_error_on_request_exception(monkeypatch):
    def fake_post(*args, **kwargs):
        raise exec_mod.requests.exceptions.RequestException("boom")

    monkeypatch.setattr(exec_mod.requests, "post", fake_post)

    out = exec_mod.execute_code_on_k8s(
        code="print('x')",
        service_ip="10.0.0.1",
        service_port="9999",
        timeout=1,
    )
    assert out["status"] == "error"
    assert "boom" in out["message"]
    assert "execution_time" in out


# -------------------------
# execution_agent_node (k8s path): no-code, reprovision, output parse branches
# -------------------------

def test_execution_agent_node_returns_failed_when_no_code():
    state = _base_state(coder_definition={"code": ""})
    new_state = exec_mod.execution_agent_node(state)

    assert new_state["execution_result"]["status"] == "failed"
    assert "No code provided for execution" in new_state["messages"][-1].content

def test_execution_agent_node_reprovisions_and_parses_csv_and_sets_file_data(monkeypatch):
    # infra not provisioned -> should call infra_agent_node()
    state = _base_state(
        coder_definition={"code": "```python\nprint('hi')\n```"},
        infrastructure_provisioned={"status": "not_provisioned", "details": []},
        data_source_location="in.csv",
        output_location="out.csv",
    )

    def fake_infra_agent_node(s):
        s = s.copy()
        s["infrastructure_provisioned"] = {
            "status": "provisioned",
            "details": [
                {
                    "step_name": "Get infra-agent-service IP/Port",
                    "status": "SUCCESS",
                    "details": {"ip": "1.2.3.4", "port": "8080"},
                }
            ],
        }
        return s

    monkeypatch.setattr(exec_mod, "infra_agent_node", fake_infra_agent_node)

    df = pd.DataFrame([{"x": 1, "y": 2}])
    csv_str = df.to_csv(index=False)
    b64 = base64.b64encode(csv_str.encode("utf-8")).decode("utf-8")

    def fake_execute_code_on_k8s(**kwargs):
        return {
            "status": "success",
            "output": f"data:text/csv;base64,{b64}",
            "output_file": {
                "filename": "out.csv",
                "content": f"data:text/csv;base64,{b64}",
                "size": len(csv_str.encode("utf-8")),
            },
        }

    monkeypatch.setattr(exec_mod, "execute_code_on_k8s", fake_execute_code_on_k8s)

    new_state = exec_mod.execution_agent_node(state)

    assert new_state["output_file_data"]["filename"] == "out.csv"
    assert new_state["output_file_data"]["content"].startswith("data:text/csv;base64,")

    # markdown table content (robust to spacing/padding differences)
    table = new_state["messages"][-1].content
    header = table.splitlines()[0] if table else ""
    assert "x" in header
    assert "y" in header


def test_execution_agent_node_success_but_non_csv_output_message(monkeypatch):
    state = _base_state(
        coder_definition={"code": "print('hi')"},
        infrastructure_provisioned={
            "status": "provisioned",
            "details": [
                {
                    "step_name": "Get infra-agent-service IP/Port",
                    "status": "SUCCESS",
                    "details": {"ip": "1.2.3.4", "port": "8080"},
                }
            ],
        },
    )

    def fake_execute_code_on_k8s(**kwargs):
        return {"status": "success", "output": "not-a-data-url"}

    monkeypatch.setattr(exec_mod, "execute_code_on_k8s", fake_execute_code_on_k8s)

    new_state = exec_mod.execution_agent_node(state)
    assert "no valid csv" in new_state["messages"][-1].content.lower()


def test_execution_agent_node_failure_reports_short_reason(monkeypatch):
    """A failed execution must give the customer ONE readable reason line —
    never the raw traceback body (that goes to the server logs)."""
    state = _base_state(
        coder_definition={"code": "print('hi')"},
        infrastructure_provisioned={
            "status": "provisioned",
            "details": [
                {
                    "step_name": "Get infra-agent-service IP/Port",
                    "status": "SUCCESS",
                    "details": {"ip": "1.2.3.4", "port": "8080"},
                }
            ],
        },
    )

    def fake_execute_code_on_k8s(**kwargs):
        return {
            "status": "error",
            "message": "bad",
            "stderr": (
                "Traceback (most recent call last):\n"
                '  File "x.py", line 1, in <module>\n'
                "TypeError: read_csv() got an unexpected keyword argument 'errors'"
            ),
        }

    monkeypatch.setattr(exec_mod, "execute_code_on_k8s", fake_execute_code_on_k8s)

    new_state = exec_mod.execution_agent_node(state)
    msg = new_state["messages"][-1].content
    assert "I couldn't complete this step" in msg
    # The real error line is surfaced…
    assert "TypeError: read_csv() got an unexpected keyword argument 'errors'" in msg
    # …but the traceback body never reaches the chat.
    assert "Traceback (most recent call last)" not in msg
    assert 'File "x.py"' not in msg


# -------------------------
# execute_code_on_local (subprocess rc handling + temp cleanup)
# -------------------------

def test_execute_code_on_local_rc0_with_stderr_is_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class _Res:
        returncode = 0
        stdout = "ok"
        stderr = "SettingWithCopyWarning: ..."

    monkeypatch.setattr(exec_mod.subprocess, "run", lambda *a, **k: _Res())

    out = exec_mod.execute_code_on_local(code="print('hi')", input_data_location=None)
    assert out["status"] == "success"
    assert out["execution_error"] is None
    assert out["returncode"] == 0
    assert "SettingWithCopyWarning" in out["execution_stderr"]
    assert not (tmp_path / "temp_script.py").exists()


def test_execute_code_on_local_nonzero_rc_is_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class _Res:
        returncode = 2
        stdout = "x"
        stderr = "Traceback..."

    monkeypatch.setattr(exec_mod.subprocess, "run", lambda *a, **k: _Res())

    out = exec_mod.execute_code_on_local(code="print('hi')", input_data_location=None)
    assert out["status"] == "error"
    assert "exit status 2" in out["execution_error"]
    assert not (tmp_path / "temp_script.py").exists()


def test_execute_code_on_local_subprocess_filenotfound_sets_error_and_cleans(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def boom(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(exec_mod.subprocess, "run", boom)

    out = exec_mod.execute_code_on_local(code="print('hi')", input_data_location=None)
    assert out["status"] == "error"
    assert "python executable not found" in out["execution_error"].lower()
    assert not (tmp_path / "temp_script.py").exists()


# -------------------------
# execution_agent_node_local (output file contract + message dedup contract)
# -------------------------

def test_execution_agent_node_local_success_when_output_file_exists(tmp_path, monkeypatch):
    out_path = tmp_path / "result.csv"
    out_path.write_text("a,b\n1,2\n", encoding="utf-8")

    # avoid actually running subprocess; node only needs output file to exist
    monkeypatch.setattr(exec_mod, "execute_code_on_local", lambda **k: {"status": "success"})

    state = _base_state(
        coder_definition={"code": "print('x')"},
        output_location=str(out_path),
        coder_raw_response="raw coder msg",
    )

    new_state = exec_mod.execution_agent_node_local(state)
    assert new_state["execution_result"]["status"] == "success"
    assert new_state["output_file_data"] is not None
    assert new_state["output_file_data"]["content"].startswith("data:text/csv;base64,")


def test_execution_agent_node_local_completed_when_no_output_file(tmp_path, monkeypatch):
    out_path = tmp_path / "missing.csv"

    monkeypatch.setattr(exec_mod, "execute_code_on_local", lambda **k: {"status": "success"})

    state = _base_state(
        coder_definition={"code": "print('x')"},
        output_location=str(out_path),
    )

    new_state = exec_mod.execution_agent_node_local(state)
    assert new_state["execution_result"]["status"] == "completed"
    assert "no output file" in new_state["execution_result"]["message"].lower()
    assert new_state["output_file_data"] is None


def test_execution_agent_node_local_dedups_coder_raw_response_and_sets_generated_code(tmp_path, monkeypatch):
    out_path = tmp_path / "result.csv"
    out_path.write_text("a,b\n1,2\n", encoding="utf-8")

    monkeypatch.setattr(exec_mod, "execute_code_on_local", lambda **k: {"status": "success"})

    state = _base_state(
        coder_definition={"code": "print('x')"},
        output_location=str(out_path),
        coder_raw_response="raw coder msg",
        messages=[],  # start empty
        generated_code=None,
    )

    # First run -> should append coder_raw_response and set generated_code
    s1 = exec_mod.execution_agent_node_local(state)
    assert s1.get("generated_code") == "print('x')"
    raw_count_1 = sum(1 for m in s1["messages"] if getattr(m, "content", None) == "raw coder msg")
    assert raw_count_1 == 1

    # Second run (starting from s1) -> should NOT add raw coder msg again
    s2 = exec_mod.execution_agent_node_local(s1)
    raw_count_2 = sum(1 for m in s2["messages"] if getattr(m, "content", None) == "raw coder msg")
    assert raw_count_2 == 1



def read_csv_best_effort(path: str, **kwargs):
    """
    Robust CSV reader:
    - tries strict decoding first (so wrong encodings raise)
    - then falls back to replace only at the end
    """
    import pandas as pd

    # quick BOM check for UTF-16
    try:
        with open(path, "rb") as f:
            head = f.read(4)
    except Exception:
        head = b""

    encodings = []
    if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
        encodings.append("utf-16")

    # try these strictly first
    encodings += ["utf-8-sig", "utf-8", "cp1252", "latin-1"]

    last_err = None

    for enc in encodings:
        try:
            # pandas >= 2 supports encoding_errors
            return pd.read_csv(path, encoding=enc, encoding_errors="strict", **kwargs)
        except TypeError:
            # older pandas: no encoding_errors param (defaults to strict anyway)
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError as e:
                last_err = e
            except Exception as e:
                last_err = e
        except UnicodeDecodeError as e:
            last_err = e
        except Exception as e:
            last_err = e

    # LAST RESORT: never crash, replace undecodable bytes
    try:
        return pd.read_csv(path, encoding="utf-8", encoding_errors="replace", **kwargs)
    except TypeError:
        return pd.read_csv(path, encoding="utf-8", **kwargs)

    # if even that fails
    raise last_err




def test_execute_code_on_local_rewrite_only_literal_paths():
    code = """
import pandas as pd
from io import StringIO
csv = "a,b\\n1,2\\n"
df1 = pd.read_csv(StringIO(csv))   # should NOT rewrite
path = "x.csv"
df2 = pd.read_csv(path)            # should NOT rewrite (variable)
df3 = pd.read_csv("real.csv")      # SHOULD rewrite
"""
    rewritten = re.sub(
        r"pd\.read_csv\(\s*([\"'][^\"']+[\"'])([^\)]*)\)",
        r"read_csv_best_effort(\1\2)",
        exec_mod._strip_markdown_code_fences(code),
        flags=re.MULTILINE,
    )
    assert "pd.read_csv(StringIO(csv))" in rewritten
    assert "pd.read_csv(path)" in rewritten
    assert 'read_csv_best_effort("real.csv")' in rewritten




def test_execute_code_on_local_generated_script_has_no_leading_indent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    captured = {}
    def fake_run(*a, **k):
        # read the created script before cleanup
        captured["script"] = (tmp_path / "temp_script.py").read_text(encoding="utf-8")
        class _Res:
            returncode = 0
            stdout = ""
            stderr = ""
        return _Res()

    monkeypatch.setattr(exec_mod.subprocess, "run", fake_run)

    exec_mod.execute_code_on_local(code="print('hi')", input_data_location=None)

    # first non-empty line must not start with spaces/tabs
    first_line = next(line for line in captured["script"].splitlines() if line.strip())
    assert not first_line.startswith((" ", "\t"))


# -------------------------
# execute_code_on_ssh (subprocess rc handling + temp cleanup)
# -------------------------

def test_execute_code_on_ssh_rc0_with_stderr_is_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class _Res:
        returncode = 0
        stdout = "ok"
        stderr = "SettingWithCopyWarning: ..."

    monkeypatch.setattr(exec_mod.subprocess, "run", lambda *a, **k: _Res())

    out = exec_mod.execute_code_on_ssh(code="print('hi')", server=CELERY_SSH_SERVER, input_data_location=tmp_path, output_location=tmp_path)
    assert out["status"] == "success"
    assert out["execution_error"] is None
    assert out["returncode"] == 0
    assert "SettingWithCopyWarning" in out["execution_stderr"]
    assert not (tmp_path / "temp_script.py").exists()


def test_execute_code_on_ssh_nonzero_rc_is_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class _Res:
        returncode = 2
        stdout = "x"
        stderr = "Traceback..."

    monkeypatch.setattr(exec_mod.subprocess, "run", lambda *a, **k: _Res())

    out = exec_mod.execute_code_on_ssh(code="print('hi')", server=CELERY_SSH_SERVER, input_data_location=tmp_path, output_location=tmp_path)
    assert out["status"] == "error"
    assert "exit status 2" in out["execution_error"]
    assert not (tmp_path / "temp_script.py").exists()


def test_execute_code_on_ssh_subprocess_filenotfound_sets_error_and_cleans(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def boom(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(exec_mod.subprocess, "run", boom)

    out = exec_mod.execute_code_on_ssh(code="print('hi')", server=CELERY_SSH_SERVER, input_data_location=tmp_path, output_location=tmp_path)
    assert out["status"] == "error"
    assert "python executable not found" in out["execution_error"].lower()
    assert not (tmp_path / "temp_script.py").exists()


# -------------------------
# execution_agent_node_ssh (output file contract + message dedup contract)
# -------------------------

def test_execution_agent_node_ssh_success_when_output_file_exists(tmp_path, monkeypatch):
    out_path = tmp_path / "result.csv"
    out_path.write_text("a,b\n1,2\n", encoding="utf-8")

    # avoid actually running subprocess; node only needs output file to exist
    monkeypatch.setattr(exec_mod, "execute_code_on_ssh", lambda **k: {"status": "success"})

    state = _base_state(
        coder_definition={"code": "print('x')"},
        data_source_location=str(tmp_path),
        output_location=str(out_path),
        coder_raw_response="raw coder msg",
    )

    new_state = exec_mod.execution_agent_node_ssh(state, CELERY_SSH_SERVER)
    assert new_state["execution_result"]["status"] == "success"
    assert new_state["output_file_data"] is not None
    assert new_state["output_file_data"]["content"].startswith("data:text/csv;base64,")


def test_execution_agent_node_ssh_completed_when_no_output_file(tmp_path, monkeypatch):
    out_path = tmp_path / "missing.csv"

    monkeypatch.setattr(exec_mod, "execute_code_on_ssh", lambda **k: {"status": "success"})

    state = _base_state(
        coder_definition={"code": "print('x')"},
        data_source_location=str(tmp_path),
        output_location=str(out_path),
    )

    new_state = exec_mod.execution_agent_node_ssh(state, CELERY_SSH_SERVER)
    assert new_state["execution_result"]["status"] == "completed"
    assert "no output file" in new_state["execution_result"]["message"].lower()
    assert new_state["output_file_data"] is None


def test_execution_agent_node_ssh_dedups_coder_raw_response_and_sets_generated_code(tmp_path, monkeypatch):
    out_path = tmp_path / "result.csv"
    out_path.write_text("a,b\n1,2\n", encoding="utf-8")

    monkeypatch.setattr(exec_mod, "execute_code_on_ssh", lambda **k: {"status": "success"})

    state = _base_state(
        coder_definition={"code": "print('x')"},
        data_source_location=str(tmp_path),
        output_location=str(out_path),
        coder_raw_response="raw coder msg",
        messages=[],  # start empty
        generated_code=None,
    )

    # First run -> should append coder_raw_response and set generated_code
    s1 = exec_mod.execution_agent_node_ssh(state, CELERY_SSH_SERVER)
    assert s1.get("generated_code") == "print('x')"
    raw_count_1 = sum(1 for m in s1["messages"] if getattr(m, "content", None) == "raw coder msg")
    assert raw_count_1 == 1

    # Second run (starting from s1) -> should NOT add raw coder msg again
    s2 = exec_mod.execution_agent_node_ssh(s1, CELERY_SSH_SERVER)
    raw_count_2 = sum(1 for m in s2["messages"] if getattr(m, "content", None) == "raw coder msg")
    assert raw_count_2 == 1

def test_execute_code_on_ssh_generated_script_has_no_leading_indent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    captured = {}
    def fake_run(*a, **k):
        # read the created script before cleanup
        captured["script"] = (tmp_path / "temp_script.py").read_text(encoding="utf-8")
        class _Res:
            returncode = 0
            stdout = ""
            stderr = ""
        return _Res()

    monkeypatch.setattr(exec_mod.subprocess, "run", fake_run)

    exec_mod.execute_code_on_ssh(code="print('hi')", server=CELERY_SSH_SERVER, input_data_location=tmp_path, output_location=tmp_path)

    # first non-empty line must not start with spaces/tabs
    first_line = next(line for line in captured["script"].splitlines() if line.strip())
    assert not first_line.startswith((" ", "\t"))
