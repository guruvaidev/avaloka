"""execution_agent_node_ssh, which copies its output back with a real `scp`.

Needs: an `scp` binary and an SSH server at CELERY_SSH_SERVERS (default
"localhost") that accepts this user. These three tests stub
`execute_code_on_ssh` but not the `subprocess.run(["scp", ...])` the node makes
afterwards (app/agents/execution_agent.py), so they open a real SSH connection
to whatever that variable names. Without `scp` they raise FileNotFoundError;
with it, the result depends on the host.

They were briefly in tests/test_execution_unit.py with the rest of that file.
They are not hermetic, so they are back behind the `integration` marker. The
honest fix is to stub the copy or to run them against an sshd container.
"""
import pandas as pd  # noqa: F401  (kept with the moved tests)
import pytest  # noqa: F401

from tests.test_execution_unit import CELERY_SSH_SERVER, _base_state, exec_mod


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


