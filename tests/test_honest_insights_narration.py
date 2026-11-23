"""W4: the web app's first look, its team story, its routing and its joins
say only what the data and the request say (web e2e, 2026-10-18).

Each test here reproduces one hand-graded defect on the code path that
produced it, and fails on the base:

* "`loan_status` leans on 'Current' (105% of rows)" -- a stratified sample's
  weighted count divided by the file's row count;
* "`Notes` is 45% empty" where 1 cell in 60 was -- "n/a", "none", "NULL"
  counted as gaps in a text column, against file_handler/null_policy.py;
* "Is `smoking_status` the outcome?" on the stroke data, where it is `stroke`;
* "what predicts a stroke?" refused as inference of a model that does not exist;
* the team story naming no finding, saying "you asked for 8" when nobody
  asked, and calling a heuristic "measured work";
* a join written as main(df_customers, df_orders) crashing with "missing 1
  required positional argument".
"""

from __future__ import annotations

import asyncio
import csv
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.api import chat_voice

FIXTURES = Path(__file__).parent / "fixtures" / "join_two_frames"


def _shares(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)% of rows", text)]


# ------------------------------------------------------------- 1. denominator
def test_a_weighted_stratified_count_is_never_a_share_above_100_percent():
    """The real hybrid stats, from a K-capped stratified sample whose weights
    over-count the top group (daft's fraction sample returns more rows than
    the cap): the weighted count exceeds the file's rows. The share comes
    from the sample's own frequency, which is the truth here."""
    daft = pytest.importorskip("daft")
    from app.agents.sampling_agent_daft import _compute_categorical_stats_hybrid

    n_rows = 1000
    base = daft.from_pydict({"loan_status": ["Current"] * 880 + ["Fully Paid"] * 120})
    # 60 "Current" rows drawn where 50 were expected: weight 880/50 = 17.6
    # gives a weighted count of 1,056 -- 105.6% of 1,000 rows.
    strat = daft.from_pydict({
        "loan_status": ["Current"] * 60 + ["Fully Paid"] * 120,
        "_weight": [880 / 50] * 60 + [1.0] * 120,
    })
    col = _compute_categorical_stats_hybrid(strat, base.collect().to_arrow(), "loan_status")
    assert col["top_values"][0]["count"] > n_rows  # the over-count this test is about

    stats = {"data_shape": {"rows": n_rows, "columns": 1}, "column_statistics": {"loan_status": col}}
    message = chat_voice.prompts_after_upload(sample_statistics=stats)[0]["message"]

    assert _shares(message), message
    assert all(s <= 100 for s in _shares(message)), message
    expected = col["top_values"][0]["frequency"]
    assert f"({expected:.0%} of rows)" in message, message


def test_a_one_pass_count_uses_its_own_total_count():
    """The Ray sampler's shape: exact counts over total_count (nulls included)."""
    stats = {"data_shape": {"rows": 5000, "columns": 1}, "column_statistics": {"grade": {
        "type": "categorical", "total_count": 200, "missing_ratio": 0.0, "n_unique": 3,
        "top_values": [{"value": "A", "count": 120}, {"value": "B", "count": 50}]}}}
    message = chat_voice.prompts_after_upload(sample_statistics=stats)[0]["message"]
    assert "(60% of rows)" in message, message


# ------------------------------------------------------------- 1. null words
def _notes_column() -> list:
    # 60 rows: 1 truly empty, 26 null-like words written as labels, 33 notes.
    words = ["n/a", "none", "NULL", "N/A", "null", "None"]
    return [None] + [words[i % len(words)] for i in range(26)] + [f"note {i}" for i in range(33)]


def test_null_word_labels_in_a_text_column_are_not_gaps_in_the_profile():
    daft = pytest.importorskip("daft")
    from app.agents.sampling_agent_daft import aggregate_statistics_from_portfolio

    notes = _notes_column()
    glucose = [None if i in (3, 9) else 100.0 + i for i in range(60)]
    df = daft.from_pydict({"Notes": notes, "Glucose": glucose,
                           "Outcome": ["Recovered", "Readmitted", "RECOVERED"] * 20})
    portfolio = {"random_baseline": df.with_column("_weight", daft.lit(1.0))}
    stats = aggregate_statistics_from_portfolio(portfolio, df)
    col = stats["column_statistics"]["Notes"]

    assert col["missing_count"] == 1, col
    assert col["missing_ratio"] == pytest.approx(1 / 60, abs=1e-4)
    stats["data_shape"] = {"rows": 60, "columns": 3}
    message = chat_voice.prompts_after_upload(sample_statistics=stats)[0]["message"]
    assert "45% empty" not in message and "<missing>" not in message, message


def test_null_words_in_a_numeric_column_are_still_gaps():
    from app.agents.sampling_agent_daft import _compute_stats_from_base

    col = _compute_stats_from_base(["98", "NA", "105", "none", "110", "87", "92", "-", "101", "99"])
    assert col["type"] == "numeric" and col["missing_count"] == 3, col


def test_the_chart_profiler_follows_the_same_rule():
    from app.agents.visualization_agent import profile_columns

    rows = [{"Notes": v} for v in _notes_column()]
    (notes,) = profile_columns(rows, ["Notes"])
    assert notes["missing_ratio"] == pytest.approx(1 / 60), notes


def _shared_cases():
    from tests.shared_cases.null_word_labels import CASE_IDS, CASES
    return pytest.mark.parametrize("case", CASES, ids=CASE_IDS)


@_shared_cases()
def test_the_profiler_counts_the_gaps_the_platform_reader_makes(tmp_path, case):
    """Parity: the profiler, handed the raw cells, finds exactly the missing
    cells file_handler/null_policy.read_csv_best_effort finds, case by case."""
    from app.agents.sampling_agent_daft import _compute_stats_from_base
    from file_handler import null_policy
    from tests.shared_cases.null_word_labels import MISSING, outcome

    path = tmp_path / "case.csv"
    path.write_text(case.csv)
    platform = outcome(null_policy.read_csv_best_effort(str(path)))
    rows = list(csv.DictReader(io.StringIO(case.csv)))
    for column, cells in platform.items():
        raw = [row[column] for row in rows]
        assert _compute_stats_from_base(raw)["missing_count"] == sum(c is MISSING for c in cells), (
            f"{case.name!r}: column {column!r}")


# ------------------------------------------------------------- 1. outcome pick
def _stroke_stats() -> dict:
    def cat(values: dict, total: int = 5110) -> dict:
        top = sorted(values.items(), key=lambda kv: -kv[1])
        return {"type": "categorical", "total_count": total, "missing_ratio": 0.0,
                "n_unique": len(values), "top_values": [{"value": k, "count": v} for k, v in top]}

    def binary(mean: float) -> dict:
        return {"type": "numeric", "n_unique": 2, "missing_ratio": 0.0, "mean": mean, "min": 0, "max": 1}

    columns = {
        "id": {"type": "numeric", "n_unique": 5110, "missing_ratio": 0.0},
        "gender": cat({"Female": 2994, "Male": 2115, "Other": 1}),
        "age": {"type": "numeric", "n_unique": 104, "missing_ratio": 0.0},
        "hypertension": binary(0.097),
        "heart_disease": binary(0.054),
        "ever_married": cat({"Yes": 3353, "No": 1757}),
        "work_type": cat({"Private": 2925, "Self-employed": 819, "children": 687,
                          "Govt_job": 657, "Never_worked": 22}),
        "Residence_type": cat({"Urban": 2596, "Rural": 2514}),
        "avg_glucose_level": {"type": "numeric", "n_unique": 3979, "missing_ratio": 0.0},
        "bmi": {"type": "numeric", "n_unique": 418, "missing_ratio": 0.039},
        "smoking_status": cat({"never smoked": 1892, "Unknown": 1544, "formerly smoked": 885,
                               "smokes": 789}),
        "stroke": binary(0.0487),
    }
    return {"data_shape": {"rows": 5110, "columns": len(columns)}, "column_statistics": columns}


def test_on_stroke_data_the_outcome_is_stroke_not_smoking_status():
    read = chat_voice.read_statistics(_stroke_stats())
    assert read.target == "stroke", read.targets
    prompt = chat_voice.prompts_after_upload(sample_statistics=_stroke_stats())[0]
    assert "`stroke`" in prompt["question"] and "smoking_status" not in prompt["question"], prompt


def test_the_column_the_goal_names_is_the_outcome():
    """The goal names `ever_married`: the next step offers a model of it, not
    of the column the upload heuristic ranked first."""
    # A run that produced rows: #545 says "no answer yet" for a run without any.
    final = {"avaloka_intent": "statistical_analysis",
             "execution_result": {"status": "success", "message": "done", "output_row_count": 2},
             "output_json": [{"ever_married": "Yes", "stroke_rate": 0.0656},
                             {"ever_married": "No", "stroke_rate": 0.0165}]}
    prompts = {p["step"]: p for p in chat_voice.prompts_after_turn(
        final, question="what drives ever_married?", sample_statistics=_stroke_stats())}
    assert "`ever_married`" in prompts["understanding"]["question"]
    labels = [r["label"] for r in prompts["result"]["quick_replies"]]
    assert "Train a model on ever_married" in labels, labels
    assert not any("stroke" in label or "smoking_status" in label for label in labels), labels


# ------------------------------------------------------------- 3. routing
def test_what_predicts_a_stroke_is_analysis_not_inference_of_a_missing_model():
    from app.agents.avaloka_agent.agent import _preconfigure_for_planner
    from app.agents.avaloka_agent.intent_classifier import _heuristic_intent, classify_intent
    from langchain_core.messages import HumanMessage

    for goal in ("what predicts a stroke?", "What drives churn?", "drivers of default"):
        result = classify_intent([HumanMessage(content=goal)])
        assert result.intent == "statistical_analysis", (goal, result)
        assert result.should_delegate()
        assert not _preconfigure_for_planner({}, result).get("enable_training")
    # Asking a model to score rows is still inference; asking for one, training.
    assert _heuristic_intent("predict churn for these customers", False).intent == "ml_inference"
    assert _heuristic_intent("what does the model predict for row 5?", False).intent == "ml_inference"
    assert _heuristic_intent("train a model to predict stroke", False).intent == "ml_training"


# ------------------------------------------------------------- 2. team story
def _stroke_like_csv(path: Path, n: int = 60_000, seed: int = 11) -> pd.DataFrame:
    """Big enough that the data sizes the fleet above the Free cap of two."""
    rng = np.random.default_rng(seed)
    age = rng.integers(1, 90, n)
    hypertension = (rng.random(n) < 0.1).astype(int)
    stroke = (rng.random(n) < 0.02 + 0.0008 * age + 0.06 * hypertension).astype(int)
    frame = pd.DataFrame({
        "age": age, "hypertension": hypertension,
        "smoking_status": rng.choice(["never smoked", "Unknown", "smokes"], n),
        "bmi": np.where(rng.random(n) < 0.04, np.nan, rng.normal(28, 6, n).round(1)),
        "stroke": stroke,
    })
    frame.to_csv(path, index=False)
    return frame


def test_the_chat_team_story_names_a_finding_and_invents_no_request(tmp_path, monkeypatch):
    """The app's own path: chat_swarm.run_mission -> swarm_service -> the
    coordinator, nobody naming a clone count, on the Free edition."""
    from app.api import chat_swarm
    from app.trace import swarm_store

    monkeypatch.setenv("AVALOKA_SWARM_OUTPUT_DIR", str(tmp_path / "swarm-out"))
    frame = _stroke_like_csv(tmp_path / "stroke.csv")
    swarm_store.registry.clear()
    try:
        mission = asyncio.run(chat_swarm.run_mission(
            "w4-user", dataset_path=str(tmp_path / "stroke.csv"),
            goal="what predicts a stroke", edition="free", timeout_s=300))
        assert mission["finished"] and mission["status"] == "done", mission["status"]
        records = mission["records"]
    finally:
        for entry in swarm_store.registry.for_user("w4-user"):
            if entry.thread is not None:
                entry.thread.join(timeout=300)
        swarm_store.registry.clear()

    message = chat_swarm.summary_prompt(records, edition="free", mission_id="m")["message"]
    requested = next(r for r in records if r["type"] == "swarm.requested")
    assert requested["limited"] and requested["derived_from"], requested  # the case at issue
    assert "you asked for" not in message, message
    assert f"the data sized the fleet at {requested['requested_clones']}" in message, message

    rate, ones = frame["stroke"].mean(), int(frame["stroke"].sum())
    assert (f"stroke is 1 in {rate:.1%} of the {len(frame):,} rows with a value ({ones:,} rows)"
            in message), message

    assert "human team" not in message and "measured work" not in message, message
    assert "rough heuristic estimate, not measured" in message, message


# ------------------------------------------------------------- 4. the join
_JOIN_CODE = '''import pandas as pd
import numpy as np

def main(df_customers, df_orders):
    df = pd.merge(df_orders, df_customers, on='customer_id', how='left')
    grouped = df.groupby('segment', as_index=False).agg(average_amount=('amount', 'mean'))
    return grouped

if __name__ == "__main__":
    df_customers = read_csv_best_effort("customers.csv")
    df_orders = read_csv_best_effort("orders.csv")
    df = main(df_customers, df_orders)
    df.to_csv("output.csv", index=False)
'''


def _truth() -> pd.DataFrame:
    customers = pd.read_csv(FIXTURES / "customers.csv")
    orders = pd.read_csv(FIXTURES / "orders.csv")
    joined = orders.merge(customers, on="customer_id", how="left")
    return (joined.groupby("segment", as_index=False).agg(average_amount=("amount", "mean"))
            .sort_values("segment").reset_index(drop=True))


def _read_map() -> dict:
    out = {}
    for name in ("customers", "orders"):
        out[f"{name}.csv"] = str(FIXTURES / f"{name}.csv")
        out[name] = str(FIXTURES / f"{name}.csv")
    return out


def test_the_join_executes_end_to_end_with_two_frames(tmp_path):
    """The code the coder wrote in the e2e run, through the real local driver."""
    from app.agents.execution_agent import execute_code_on_local

    output = tmp_path / "output.csv"
    result = execute_code_on_local(
        code=_JOIN_CODE, input_data_location=str(FIXTURES / "customers.csv"),
        output_location=str(output), read_path_map=_read_map())
    assert result.get("status") != "error", result
    got = pd.read_csv(output).sort_values("segment").reset_index(drop=True)
    pd.testing.assert_frame_equal(got, _truth(), check_dtype=False)


def test_the_validator_sample_run_hands_main_both_frames():
    from app.agents.validator import execute_code_node

    datasets = [{"alias": n, "filename": f"{n}.csv", "data_source_location": str(FIXTURES / f"{n}.csv")}
                for n in ("customers", "orders")]
    state = {
        "generated_code": _JOIN_CODE,
        "sample_data": (FIXTURES / "customers.csv").read_text(),
        "data_source_location": str(FIXTURES / "customers.csv"),
        "datasets_context": datasets,
    }
    out = execute_code_node(state)
    assert out["execution_error"] is None, out["execution_error"]
    got = pd.DataFrame(out["execution_output_data"]).sort_values("segment").reset_index(drop=True)
    pd.testing.assert_frame_equal(got, _truth(), check_dtype=False)


def test_a_parameter_that_names_no_dataset_is_an_error_not_a_guess():
    from app.agents.main_frames import bind_main_frames

    def main(df_customers, df_invoices):
        return df_customers

    with pytest.raises(TypeError, match="'df_invoices' names no dataset"):
        bind_main_frames(main, "primary", _read_map(), pd.read_csv,
                         primary_source=str(FIXTURES / "customers.csv"))
    # One parameter is the primary frame, exactly as before.
    assert bind_main_frames(lambda df: df, "primary", _read_map(), pd.read_csv) == ["primary"]
