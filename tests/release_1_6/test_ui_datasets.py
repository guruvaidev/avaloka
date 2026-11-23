"""1.6 release suite: upload, profile, Auto Insights and chat threads (plan sheet 1).

The shared server harness (tests/server_harness.py) stubs the sampler so most
server tests never look at real numbers. These tests put the REAL portfolio
sampler back on the upload path, so the profile the browser receives is
compared against ground truth computed from the uploaded file itself.
Auto Insights run with no model key, i.e. the deterministic chart selector.
"""
from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, List

import pandas as pd
import pytest

import app.api.server as server
from app.agents.sampling_agent_daft import sample_with_profiling as real_sample_with_profiling
from tests.server_harness import client, make_auth_headers  # noqa: F401  (pytest fixture)


def _ground_truth_frame(n: int = 60) -> pd.DataFrame:
    """10 columns, three dtypes, nulls at known positions."""
    regions = ["north", "south", "east", "west"]
    frame = pd.DataFrame({
        "order_no": [f"ord-{i:04d}" for i in range(n)],
        "region": [regions[i % 4] for i in range(n)],
        "channel": ["web" if i % 3 else "store" for i in range(n)],
        "units": [(i * 7) % 23 + 1 for i in range(n)],
        "unit_price": [round(5.0 + (i % 11) * 1.75, 2) for i in range(n)],
        "discount": [round((i % 5) * 0.05, 2) for i in range(n)],
        "rating": [float((i % 5) + 1) for i in range(n)],
        "returns": [i % 4 for i in range(n)],
        "segment": ["consumer" if i % 2 else "business" for i in range(n)],
        "weight_kg": [round(0.5 + (i % 9) * 0.3, 2) for i in range(n)],
    })
    frame.loc[[3, 11, 19, 27, 35, 43], "rating"] = None          # 6 / n missing
    frame.loc[[5, 25, 45], "segment"] = None                     # 3 / n missing
    return frame


@pytest.fixture
def real_sampler(monkeypatch):
    monkeypatch.setattr(server, "sample_with_profiling", real_sample_with_profiling, raising=False)


def _upload(client, name: str, payload: bytes, content_type: str, user: str) -> Dict[str, Any]:
    resp = client.post("/api/upload", files={"file": (name, payload, content_type)},
                       headers=make_auth_headers(user))
    assert resp.status_code == 200, resp.text
    return resp.json()


# --------------------------------------------------------------------------- #
# UI-01: the profile card's numbers are the file's numbers.
# --------------------------------------------------------------------------- #

def test_ui01_csv_upload_profile_matches_ground_truth(client, real_sampler):
    """UI-01: a CSV upload under the size limit returns a profile whose row count, column count, column types and missing % equal the file's ground truth."""
    frame = _ground_truth_frame()
    out = _upload(client, "orders.csv", frame.to_csv(index=False).encode(), "text/csv", "ui01-user")

    stats = out["sample_statistics"]
    assert stats["data_shape"] == {"rows": len(frame), "columns": frame.shape[1]}
    assert list(out["schema"].keys()) == list(frame.columns)

    numeric = {"units", "unit_price", "discount", "rating", "returns", "weight_kg"}
    text = set(frame.columns) - numeric
    for col in numeric:
        assert stats["column_statistics"][col]["type"] == "numeric", (col, stats["column_statistics"][col]["type"])
    for col in text:
        assert stats["column_statistics"][col]["type"] != "numeric", col

    truth_missing = frame.isna().mean()
    for col in frame.columns:
        got = stats["column_statistics"][col].get("missing_ratio", 0.0)
        assert got == pytest.approx(float(truth_missing[col]), abs=1e-9), (col, got, truth_missing[col])


# --------------------------------------------------------------------------- #
# UI-02: Excel, JSON and Parquet register with the same shape as the CSV.
# --------------------------------------------------------------------------- #

def _encode(frame: pd.DataFrame, fmt: str) -> tuple[str, bytes, str]:
    buf = io.BytesIO()
    if fmt == "csv":
        return "same.csv", frame.to_csv(index=False).encode(), "text/csv"
    if fmt == "json":
        return "same.json", frame.to_json(orient="records", lines=True).encode(), "application/json"
    if fmt == "parquet":
        frame.to_parquet(buf, index=False)
        return "same.parquet", buf.getvalue(), "application/octet-stream"
    if fmt == "xlsx":
        frame.to_excel(buf, index=False, sheet_name="data")
        return "same.xlsx", buf.getvalue(), \
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    raise AssertionError(fmt)


@pytest.mark.parametrize("fmt", ["xlsx", "json", "parquet"])
def test_ui02_every_format_registers_with_the_csv_shape(client, real_sampler, fmt):
    """UI-02: Excel, JSON and Parquet uploads are accepted and their row/column counts and columns match the same data uploaded as CSV."""
    frame = _ground_truth_frame()
    csv_out = _upload(client, *_encode(frame, "csv"), user=f"ui02-{fmt}")
    other = _upload(client, *_encode(frame, fmt), user=f"ui02-{fmt}")

    assert other["dataset_id"] != csv_out["dataset_id"]
    assert other["sample_statistics"]["data_shape"] == csv_out["sample_statistics"]["data_shape"] \
        == {"rows": len(frame), "columns": frame.shape[1]}
    assert sorted(other["schema"]) == sorted(csv_out["schema"]) == sorted(frame.columns)


# --------------------------------------------------------------------------- #
# UI-06: Auto Insights picks 3-5 charts, of types that suit the columns.
# --------------------------------------------------------------------------- #

def _insight_rows(n: int = 240) -> List[Dict[str, Any]]:
    rows = []
    products = ["alpha", "beta", "gamma", "delta", "epsilon"]
    regions = ["north", "south", "east", "west"]
    for i in range(n):
        price = 10 + (i % 17) * 3.5
        qty = 1 + (i * 7) % 13
        rows.append({
            "order_id": 100000 + i,
            "order_date": f"2025-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}",
            "region": regions[i % 4],
            "product": products[(i * 3) % 5],
            "price": price,
            "quantity": qty,
            "revenue": round(price * qty, 2),
            "delivery_days": 1 + (i * 5) % 9,
        })
    return rows


# Owner decision 2026-10-31 on the plan's "between 3 and 5": 3-5 charts plus an
# optional summary pie. The selector caps at MAX_TOTAL_CHARTS = 5 and
# _build_extra_pie_chart adds at most one pie on top (visualization_agent.py).
def test_ui06_auto_insights_charts_unprompted_and_typed_to_columns(monkeypatch):
    """UI-06: with no prompt, Auto Insights produces 3-5 non-pie charts plus at most one summary pie, typed to the columns (histograms on measures, a line over the date, bars/pies over categories) and never on an identifier."""
    from app.agents.visualization_agent import build_visualization_config_from_sample

    for key in ("OPENROUTER_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    rows = _insight_rows()
    config = build_visualization_config_from_sample(dataset_id="ui06", sample_rows=rows)

    charts = config["charts"]
    kinds = [c["type"] for c in charts]
    non_pie = [k for k in kinds if k != "pie"]
    assert 3 <= len(non_pie) <= 5, kinds
    assert kinds.count("pie") <= 1, kinds
    measures = {"price", "quantity", "revenue", "delivery_days"}
    categories = {"region", "product"}
    for chart in charts:
        enc = chart.get("encodings") or {}
        fields = {v.get("field") for v in enc.values() if isinstance(v, dict) and v.get("field")}
        assert "order_id" not in fields, chart["id"]
        kind = chart["type"]
        if kind == "histogram":
            assert (enc.get("x") or {}).get("field") in measures, chart["id"]
        elif kind == "scatter":
            assert {(enc.get("x") or {}).get("field"), (enc.get("y") or {}).get("field")} <= measures, chart["id"]
        elif kind == "line":
            assert (enc.get("x") or {}).get("field") == "order_date", chart["id"]
        elif kind == "bar":
            assert (enc.get("x") or {}).get("field") in categories, chart["id"]
        elif kind == "pie":
            assert (enc.get("color") or {}).get("field") in categories, chart["id"]
        else:
            raise AssertionError(f"unexpected chart type {kind!r} for {chart['id']}")


# --------------------------------------------------------------------------- #
# UI-10: each thread keeps its own history, and it survives a restart.
# --------------------------------------------------------------------------- #

def test_ui10_two_threads_keep_separate_histories_across_a_restart(client, monkeypatch):
    """UI-10: one chat thread per file; each thread's history holds only its own messages and is re-fetchable after the API process restarts."""
    import app.api.helpers as helpers

    # Real write-through history (the harness no-ops it) against the fake Redis.
    monkeypatch.setattr(server, "hydrate_thread_history", helpers.hydrate_thread_history, raising=False)
    monkeypatch.setattr(server, "persist_thread_history", helpers.persist_thread_history, raising=False)
    monkeypatch.setattr(server, "read_thread_msgs", helpers.read_thread_msgs, raising=False)

    user = "ui10-user"
    a = _upload(client, "first.csv", b"a,b\n1,2\n3,4\n", "text/csv", user)
    b = _upload(client, "second.csv", b"x,y\n5,6\n7,8\n", "text/csv", user)
    assert a["thread_id"] != b["thread_id"]

    for up, text in ((a, "question about the first file"), (b, "question about the second file")):
        resp = client.post(f"/threads/{up['thread_id']}/messages",
                           json={"content": text, "metadata": {"dataset_id": up["dataset_id"]}},
                           headers=make_auth_headers(user))
        assert resp.status_code == 200, resp.text

    server.THREAD_META.clear()  # a restart: the in-process cache is gone

    def history(thread_id: str) -> str:
        resp = client.get(f"/threads/{thread_id}/messages", headers=make_auth_headers(user))
        assert resp.status_code == 200, resp.text
        return json.dumps(resp.json()["messages"])

    first, second = history(a["thread_id"]), history(b["thread_id"])
    assert "question about the first file" in first
    assert "question about the second file" not in first
    assert "question about the second file" in second
    assert "question about the first file" not in second
