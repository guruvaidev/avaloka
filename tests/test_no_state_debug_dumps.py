"""
Graph state must never be logged wholesale.

Bug: plan_etl_job carried six ``logger.debug(f"...STATE: {state}")`` calls.
Graph state can hold decrypted database passwords (``dta_database_registry``),
message text and sampled rows, so with AVALOKA_LOG_LEVEL=DEBUG every planner
turn wrote them to the logs. The planner now logs state KEY NAMES only.

Covers:
- Behavioural: the incoming, stub and fallback planner paths at DEBUG never
  emit a planted synthetic password.
- Static: no logger call in app/ or avaloka/ formats graph state, a registry
  or a credentials object wholesale (f-string, %-args or str()/repr()).
"""
import ast
import logging
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from app.agents.planner import plan_etl_job
from app.graph.etl_state import ETLState

SYNTH_PW = "ZZ-SYNTH-PW-1"
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _debug_level(monkeypatch, caplog):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("AVALOKA_LOG_LEVEL", "DEBUG")
    caplog.set_level(logging.DEBUG)
    # server.py pins app.* loggers to AVALOKA_LOG_LEVEL at import; in a full
    # run that import has already happened, so set the planner's level too.
    caplog.set_level(logging.DEBUG, logger="app.agents.planner")
    yield


def _state(prompt: str, **extra) -> ETLState:
    state = ETLState(
        messages=[HumanMessage(content=prompt)],
        planner_definition={},
        session_id="debug-dump-test",
        dta_database_registry={
            "warehouse": {
                "db_type": "postgres",
                "host": "db.invalid",
                "user": "svc",
                "password": SYNTH_PW,
            }
        },
    )
    state.update(extra)
    return state


def _assert_no_leak(caplog):
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert caplog.records, "planner emitted no log records; the DEBUG path did not run"
    assert SYNTH_PW not in text, "a registry password reached the logs"


def test_incoming_state_is_not_dumped(caplog):
    plan_etl_job(_state("count rows in warehouse", skip_to_execution=True))
    assert any("INCOMING STATE" in r.getMessage() for r in caplog.records)
    _assert_no_leak(caplog)


@patch("app.agents.planner.llm", None)
@patch("app.agents.planner._fallback_planner_action_without_llm", return_value=None)
def test_stub_outgoing_state_is_not_dumped(_fb, caplog):
    plan_etl_job(_state("count the rows in the orders table by status"))
    assert any("OUTGOING STATE (stub)" in r.getMessage() for r in caplog.records)
    _assert_no_leak(caplog)


@patch("app.agents.planner.llm")
def test_fallback_outgoing_state_is_not_dumped(mock_llm, caplog):
    mock_llm.invoke.side_effect = RuntimeError("provider down")
    mock_llm.bind_tools.return_value = mock_llm
    plan_etl_job(_state("count the rows in the orders table by status"))
    assert any("OUTGOING STATE" in r.getMessage() for r in caplog.records)
    _assert_no_leak(caplog)


# ── Static ───────────────────────────────────────────────────────────────────

_LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}
_SECRETISH = re.compile(r"(^|_)(registry|creds|credentials?)$", re.I)
_STATEISH = re.compile(r"(^|_)state$")


def _is_log_call(node: ast.Call) -> bool:
    f = node.func
    if not isinstance(f, ast.Attribute) or f.attr not in _LOG_METHODS:
        return False
    owner = f.value
    name = owner.id if isinstance(owner, ast.Name) else getattr(owner, "attr", "")
    return "log" in name.lower()


def _graph_state_names(fn) -> set:
    """Bare ``state`` names that hold a mapping: parameters, or locals built
    from a dict / ETLState(...). A local such as ``state = resp.get("state")``
    is a scalar status string and is fine to log. ``*_state`` names
    (final_state, input_state, ...) are always treated as graph state."""
    names = set()
    if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        names = {a.arg for a in fn.args.args + fn.args.kwonlyargs if _STATEISH.search(a.arg)}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Dict, ast.Call)):
            if isinstance(node.value, ast.Call):
                callee = node.value.func
                cname = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", "")
                if not (cname.endswith("State") or cname.endswith("_state") or cname == "dict"):
                    continue
            for t in node.targets:
                if isinstance(t, ast.Name) and _STATEISH.search(t.id):
                    names.add(t.id)
    return names


def _wholesale(expr, state_names):
    if isinstance(expr, ast.Call) and expr.args:
        callee = expr.func
        cname = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", "")
        if cname in {"str", "repr", "dumps", "pformat", "dict", "vars"}:
            return _wholesale(expr.args[0], state_names)
    if isinstance(expr, ast.Name) and (
        expr.id in state_names or expr.id.endswith("_state") or _SECRETISH.search(expr.id)
    ):
        return expr.id
    return None


def _violations(path: Path, root: Path = REPO):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    scopes = [(fn, _graph_state_names(fn)) for fn in fns] + [(tree, _graph_state_names(tree))]
    seen = set()
    for scope, names in scopes:
        for node in ast.walk(scope):
            if not (isinstance(node, ast.Call) and _is_log_call(node)) or id(node) in seen:
                continue
            exprs = []
            for arg in node.args:
                if isinstance(arg, ast.JoinedStr):
                    exprs += [v.value for v in arg.values if isinstance(v, ast.FormattedValue)]
                else:
                    exprs.append(arg)
            hits = [h for h in (_wholesale(e, names) for e in exprs) if h]
            if hits:
                seen.add(id(node))
                out.append(f"{path.relative_to(root)}:{node.lineno} logs {hits}")
    return out


def test_no_logger_call_formats_state_or_credentials_wholesale():
    offenders = []
    for root in ("app", "avaloka"):
        for path in sorted((REPO / root).rglob("*.py")):
            offenders += _violations(path)
    assert not offenders, "wholesale state/credential logging:\n" + "\n".join(offenders)


def test_static_check_bites_on_the_original_pattern(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import logging\nlogger = logging.getLogger(__name__)\n"
        "def node(state):\n"
        "    logger.debug(f'Planner OUTGOING STATE: {state}')\n"
        "    logger.debug('state=%s', state)\n"
        "    logger.info('keys=%s', sorted(state))\n"
        "def poll(resp):\n"
        "    state = resp.get('state')\n"
        "    logger.debug('cluster state=%s', state)\n"
        "def task(x):\n"
        "    final_state = run(x)\n"
        "    logger.info(f'Final state: {final_state}')\n"
    )
    found = _violations(sample, root=tmp_path)
    assert [f.split(":")[1].split()[0] for f in found] == ["4", "5", "12"], found
