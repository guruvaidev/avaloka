"""rc12's "what are the strongest predictors of stroke?" (groq, clinic DB), routed.

All three rc12 threads (results/pred/p1..p3) sent the question to the coder,
and the reply's figures were the model's, not the data's:

- p1 shown: "BMI, ever married, smoking status, and residence type have much
  smaller effects, the last barely 0.003" (ever_married's eta is 0.108);
- p2 shown: "work_type_Self-employed (0.32) and heart_disease (0.27) are also
  strong" (raw logistic coefficients on unscaled dummies);
- p3 shown: "avg_glucose_level (~0.132) and heart_disease (~0.129)",
  "Hypertension follows at ~0.093", "All other factors ... less than 0.001";
  and withheld "Age is the strongest predictor of stroke (importance ~0.245)".

The question now goes to the shared driver ranking and is said from a
template: no coder, no LLM, every figure the table's. These run the real chat
route on an upload thread and on a database-table thread; the graph (planner,
coder, executor, narrator) raises if it is reached. An explicit model request
still reaches it. No network, no LLM.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

import app.api.server as server
from avaloka.goal_analysis import analyze_goal, driver_ranking
from file_handler.null_policy import read_csv_best_effort

try:
    from tests.server_harness import (  # noqa: F401  (fixtures are used by name)
        FakeGraph, SESSIONS, client, make_auth_headers, _quiet_send)
    from tests.test_db_sampling_honesty import mcp_runner
    from tests.test_driver_ranking_scale_free import stroke_frame
except ImportError:  # tests/ on sys.path
    from server_harness import FakeGraph, SESSIONS, client, make_auth_headers, _quiet_send  # noqa: F401
    from test_db_sampling_honesty import mcp_runner
    from test_driver_ranking_scale_free import stroke_frame

#: The question all three rc12 threads asked, verbatim.
QUESTION = "what are the strongest predictors of stroke?"
RC12 = {"p1": QUESTION, "p2": QUESTION, "p3": QUESTION}
#: The other wordings the route answers.
WORDINGS = ["what predicts stroke?", "what drives stroke?", "risk factors for stroke",
            "which factors affect stroke?", "strongest drivers of stroke"]
MODEL_ASK = "train a random forest and show feature importance for stroke"


def patients() -> pd.DataFrame:
    """The stroke file's shape, with the Kaggle file's scattered integer ``id``
    (a row identifier by IDENTIFIER_RULE, never a predictor)."""
    frame = stroke_frame()
    ids = np.random.default_rng(3).choice(np.arange(1, 72944), len(frame), replace=False)
    frame.insert(0, "id", ids)
    return frame


class _MustNotRunGraph(FakeGraph):
    def invoke(self, state_in, config=None):
        raise AssertionError("the coder pipeline must not run for a drivers question")


class _RecordingGraph(FakeGraph):
    def __init__(self):
        self.calls: List[str] = []

    def invoke(self, state_in, config=None):
        last = state_in["messages"][-1]
        self.calls.append(getattr(last, "content", str(last)))
        return super().invoke(state_in, config)


def _upload_thread(client, user: str, tmp_path: Path) -> Dict[str, Any]:
    csv = tmp_path / "stroke.csv"
    patients().to_csv(csv, index=False)
    resp = client.post("/api/upload", files={"file": ("stroke.csv", csv.read_bytes(), "text/csv")},
                       data={}, headers=make_auth_headers(user))
    assert resp.status_code == 200, resp.text
    return resp.json()


def _db_thread(client, monkeypatch, user: str, tmp_path: Path, cap: int | None = None) -> Dict[str, Any]:
    if cap is not None:
        monkeypatch.setattr(server, "MAX_DB_SAMPLE_ROWS", cap, raising=False)
    path = tmp_path / "clinic.db"
    con = sqlite3.connect(path)
    patients().to_sql("patients", con, index=False)
    con.close()
    run = mcp_runner(path, [])

    async def _mcp(customer_id, api_key, sql):
        return await run(sql)

    monkeypatch.setattr(server, "_run_mcp_db_query", _mcp, raising=False)
    resp = client.post("/api/database/tables-to-analysis",
                       json={"tables": ["patients"], "customer_id": "clinic", "api_key": "k-clinic"},
                       headers=make_auth_headers(user))
    assert resp.status_code == 200, resp.text
    out = resp.json()
    return {"dataset_id": out["datasets"][0]["dataset_id"], "thread_id": out["thread_id"],
            "session_id": out["session_id"]}


def _ask(client, thread: Dict[str, Any], question: str, user: str) -> Dict[str, Any]:
    resp = client.post(f"/threads/{thread['thread_id']}/messages",
                       json={"content": question, "metadata": {"dataset_id": thread["dataset_id"]}},
                       headers=make_auth_headers(user))
    assert resp.status_code == 200, resp.text
    return resp.json()


def _table(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    tables = [t for t in body.get("output_tables") or [] if t.get("action_ids") == ["drivers_route"]]
    assert tables, f"no drivers table: the turn did not take the deterministic route ({_text(body)[:300]!r})"
    return tables[0]["rows"]


def _text(body: Dict[str, Any]) -> str:
    return [m for m in body["messages"] if m["role"] == "assistant"][-1]["content"]


def _session_rows(dataset_id: str) -> pd.DataFrame:
    """The rows the thread holds, read as the analysis reads them."""
    sess = next(s for s in SESSIONS.values() if s.get("dataset_id") == dataset_id)
    path = sess.get("work_local_input") or sess.get("sample_local_input")
    return read_csv_best_effort(path)


def _truth(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """driver_ranking's table: the five strongest, leakage set aside."""
    _stats, drivers, _skipped = driver_ranking(frame, "stroke")
    return [d for d in drivers if not d.get("leakage")][:5]


def _numbers(text: str) -> List[float]:
    return [float(x) for x in re.findall(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?", text.replace(",", ""))]


def _assert_answer(body: Dict[str, Any], frame: pd.DataFrame) -> None:
    text = _text(body)
    truth = _truth(frame)
    rows = _table(body)
    # The answer's table is driver_ranking's table.
    assert [(r["column"], r["association"]) for r in rows] == [
        (d["column"], round(d["strength"], 3)) for d in truth], (rows, truth)
    # Age is first, in the sentence and the table.
    assert truth[0]["column"] == "age"
    assert text.startswith(f"age has the strongest association with stroke (r = {abs(truth[0]['strength']):.3f})"), text
    # Every figure in the text is the table's: a strength, a figure of a
    # row's own detail, or the row count of the basis line.
    allowed = {round(abs(d["strength"]), 3) for d in truth}
    for d in truth:
        allowed |= {abs(v) for v in _numbers(d.get("detail") or "")}
    allowed |= {float(len(frame)), float(5110)}
    allowed |= set(range(1, 6))       # the table's rank column
    stray = [v for v in _numbers(text) if round(abs(v), 3) not in allowed and abs(v) not in allowed]
    assert not stray, (stray, text)
    for d in truth:
        assert f"{abs(d['strength']):.3f}" in text, (d, text)
    married = next((d for d in truth if d["column"] == "ever_married"), None)
    if married is not None:
        assert f"ever_married (eta = {married['strength']:.3f})" in text, text
    # rc12's false figures are gone.
    for false in ("0.003", "0.32", "0.27", "0.093", "less than 0.001"):
        assert false not in text, (false, text)
    assert "association" in text and "does not control for the other columns" in text
    assert "importance" not in text.lower()
    # No identifier, in the text or the table.
    assert not re.search(r"(?<![\w])id(?![\w])", text), text
    assert "id" not in [r["column"] for r in rows]


@pytest.mark.parametrize("label", sorted(RC12))
def test_rc12_question_on_an_upload_thread_is_answered_from_the_ranking(client, monkeypatch, tmp_path, label):
    _quiet_send(monkeypatch)
    monkeypatch.setattr(server, "GRAPH", _MustNotRunGraph(), raising=False)
    thread = _upload_thread(client, f"drv-up-{label}", tmp_path)
    body = _ask(client, thread, RC12[label], f"drv-up-{label}")
    frame = _session_rows(thread["dataset_id"])
    _assert_answer(body, frame)
    # Deterministic: asked again, the same words.
    assert _text(_ask(client, thread, RC12[label], f"drv-up-{label}")) == _text(body)


@pytest.mark.parametrize("label", sorted(RC12))
def test_rc12_question_on_a_db_thread_is_answered_from_the_ranking(client, monkeypatch, tmp_path, label):
    _quiet_send(monkeypatch)
    monkeypatch.setattr(server, "GRAPH", _MustNotRunGraph(), raising=False)
    thread = _db_thread(client, monkeypatch, f"drv-db-{label}", tmp_path)
    body = _ask(client, thread, RC12[label], f"drv-db-{label}")
    frame = _session_rows(thread["dataset_id"])
    assert len(frame) == 5110
    _assert_answer(body, frame)
    assert "Computed over all 5,110 rows of the patients table." in _text(body)


def test_a_sampled_db_thread_keeps_its_n_of_m_basis_line(client, monkeypatch, tmp_path):
    _quiet_send(monkeypatch)
    monkeypatch.setattr(server, "GRAPH", _MustNotRunGraph(), raising=False)
    thread = _db_thread(client, monkeypatch, "drv-db-sampled", tmp_path, cap=2000)
    body = _ask(client, thread, QUESTION, "drv-db-sampled")
    frame = _session_rows(thread["dataset_id"])
    assert len(frame) == 2000
    rows = _table(body)
    assert [(r["column"], r["association"]) for r in rows] == [
        (d["column"], round(d["strength"], 3)) for d in _truth(frame)]
    assert "a random sample of 2,000 of 5,110 rows" in _text(body)
    assert "sample estimate" in _text(body)


@pytest.mark.parametrize("question", WORDINGS)
def test_every_drivers_wording_takes_the_route(client, monkeypatch, tmp_path, question):
    _quiet_send(monkeypatch)
    monkeypatch.setattr(server, "GRAPH", _MustNotRunGraph(), raising=False)
    thread = _upload_thread(client, "drv-words", tmp_path)
    body = _ask(client, thread, question, "drv-words")
    assert _table(body)[0]["column"] == "age"
    assert "importance" not in _text(body).lower()


@pytest.mark.parametrize("thread_kind", ["upload", "db"])
def test_an_explicit_model_request_still_goes_to_the_coder(client, monkeypatch, tmp_path, thread_kind):
    _quiet_send(monkeypatch)
    graph = _RecordingGraph()
    monkeypatch.setattr(server, "GRAPH", graph, raising=False)
    thread = (_upload_thread(client, "drv-model", tmp_path) if thread_kind == "upload"
              else _db_thread(client, monkeypatch, "drv-model", tmp_path))
    body = _ask(client, thread, MODEL_ASK, "drv-model")
    assert graph.calls and MODEL_ASK in graph.calls[-1]
    assert not any(t.get("action_ids") == ["drivers_route"] for t in body.get("output_tables") or [])


#: The routing rule, as a table: (question, deterministic route?).
RULE = [
    (QUESTION, True), *[(w, True) for w in WORDINGS],
    ("what are the most important factors for stroke?", True),     # ambiguous: no model named
    ("which variables influence stroke the most?", True),
    (MODEL_ASK, False),
    ("train a model and tell me feature importance for stroke", False),
    ("fit a logistic regression for stroke and show the coefficients", False),
    ("what are the random forest importances of the predictors of stroke?", False),
    ("what is the stroke rate for patients with hypertension?", False),  # not a drivers question
]


@pytest.mark.parametrize("question,routed", RULE)
def test_the_routing_rule(question, routed):
    from app.agents.drivers_route import is_drivers_question

    assert is_drivers_question(question) is routed


def test_the_cli_and_the_web_give_the_same_table(client, monkeypatch, tmp_path):
    """The CLI's drivers answer (analyze_goal, as `avaloka analyze` runs it)
    and the web route agree, column for column and figure for figure."""
    _quiet_send(monkeypatch)
    monkeypatch.setattr(server, "GRAPH", _MustNotRunGraph(), raising=False)
    thread = _upload_thread(client, "drv-parity", tmp_path)
    web = _table(_ask(client, thread, QUESTION, "drv-parity"))
    frame = _session_rows(thread["dataset_id"])
    cli = analyze_goal(frame, QUESTION, {})
    assert cli["kind"] == "drivers" and cli["status"] == "answered"
    listed = [f.split(":")[0] for f in cli["findings"][1:6]]
    cli_rows = [(d["column"], round(d["strength"], 3)) for d in cli["metrics"]["drivers"] if d["column"] in listed]
    assert [(r["column"], r["association"]) for r in web] == cli_rows
    # And the CLI's own sentence about the top column is the web's first clause.
    label = next(f for f in cli["findings"] if "has the strongest association" in f)
    assert _text(_ask(client, thread, QUESTION, "drv-parity")).startswith(label.split(";")[0])
