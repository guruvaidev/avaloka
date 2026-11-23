"""The swarm's driver ranking comes from the whole dataset, not one shard.

The web re-run asked the swarm "what drives stroke" of a file sorted with
every stroke first. The shards are contiguous row ranges, so one clone held
none of them ("stroke holds 0 of 2,555 rows") and the fleet's answer listed
no driver at all, on both models. The ranking is now computed from the
clones' merged sufficient statistics (co-moments for Pearson and
point-biserial r, per-group moments for eta, contingency counts for
Cramer's V), so it is the ranking one pass over every row gives: the same
columns, in the same order, with the same values.

Fields known only after the outcome (a loan's settlement, its last payment,
what was received on it) are set aside as likely leakage by name and timing
words, whatever their strength, and the answer says so.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from avaloka import goal_merge
from avaloka.coordinator import MainAvaloka
from avaloka.goal_analysis import analyze_goal
from avaloka.mission.budget import Budget
from avaloka.mission.context import MissionContext, MissionKind
from avaloka.missions import run_analyze
from avaloka.swarm_limits import UNLIMITED

GOAL = "what drives stroke"


@pytest.fixture(autouse=True)
def _no_edition(monkeypatch):
    monkeypatch.delenv("AVALOKA_EDITION", raising=False)


def _stroke_frame(n: int = 1600, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(1, 90, n).astype(float)
    hypertension = (rng.random(n) < 0.08 + age / 400).astype(int)
    glucose = rng.normal(105, 30, n).round(2)
    logit = -5.2 + 0.045 * age + 0.9 * hypertension + 0.006 * (glucose - 105)
    stroke = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    bmi = rng.normal(28, 6, n).round(1)
    bmi[rng.random(n) < 0.04] = np.nan
    frame = pd.DataFrame({
        "id": np.arange(10_000, 10_000 + n),
        "gender": rng.choice(["Female", "Male"], n),
        "age": age,
        "hypertension": hypertension,
        "ever_married": np.where(age + rng.normal(0, 15, n) > 30, "Yes", "No"),
        "work_type": rng.choice(["Private", "Self-employed", "children", "Govt_job"], n),
        "avg_glucose_level": glucose,
        "bmi": bmi,
        "smoking_status": rng.choice(["never smoked", "Unknown", "formerly smoked", "smokes"], n),
        "stroke": stroke,
    })
    # Sorted as the Kaggle file is: every stroke first.
    return frame.sort_values("stroke", ascending=False, kind="stable").reset_index(drop=True)


@pytest.fixture(scope="module")
def stroke_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("stroke") / "stroke_sorted.csv"
    _stroke_frame().to_csv(path, index=False)
    return path


def _single_pass(source, out):
    ctx = MissionContext(kind=MissionKind.ANALYZE, goal=GOAL, data_source=str(source),
                         output_dir=out, budget=Budget())
    run_analyze(ctx)
    return json.loads((out / "goal_analysis.json").read_text())


def _fleet(source, out, workers=2):
    return MainAvaloka(swarm_limits=UNLIMITED).coordinate(
        str(source), goal=GOAL, output_dir=out, workers=workers, kind=MissionKind.ANALYZE)


def _ranking(drivers):
    return [(d["column"], d["measure"], d["strength"]) for d in drivers]


# --------------------------------------------------- (a) merged = single pass
def test_the_fixture_puts_every_stroke_in_one_shard(stroke_csv):
    """The bite's precondition: the second clone holds no positive."""
    frame = pd.read_csv(stroke_csv)
    first, second = np.array_split(frame["stroke"].to_numpy(), 2)
    assert first.sum() > 30 and second.sum() == 0


def test_the_swarm_drivers_equal_one_pass_over_the_whole_file(stroke_csv, tmp_path):
    single = _single_pass(stroke_csv, tmp_path / "single")
    cr = _fleet(stroke_csv, tmp_path / "fleet")
    answer = cr.answer
    assert single["status"] == "answered" and single["kind"] == "drivers"
    assert answer["merge"] == "sufficient statistics", answer["headline"]
    assert answer["status"] == "answered" and answer["exact"] is True, answer
    expected = _ranking(single["metrics"]["drivers"])
    assert len(expected) >= 5
    assert _ranking(answer["metrics"]["drivers"]) == expected
    # The details either side of the outcome are the whole file's too.
    assert [d.get("detail") for d in answer["metrics"]["drivers"]] == \
        [d.get("detail") for d in single["metrics"]["drivers"]]
    # And the listed lines are the single pass's lines, strongest first.
    assert answer["findings"][:6] == single["findings"][:6]
    assert answer["findings"][0].startswith("What moves with stroke, strongest first")
    lead = expected[0][0]
    assert lead in answer["headline"]


def test_the_merged_values_are_the_textbook_statistics(stroke_csv, tmp_path):
    """Independent of the code under test: point-biserial r with numpy for
    the quantities and the correlation ratio by hand, on every row."""
    frame = pd.read_csv(stroke_csv)
    answer = _fleet(stroke_csv, tmp_path / "fleet").answer
    got = {d["column"]: d["strength"] for d in answer["metrics"]["drivers"]}
    for col in ("age", "avg_glucose_level", "bmi", "hypertension"):
        both = frame[[col, "stroke"]].dropna()
        r = float(np.corrcoef(both[col], both["stroke"])[0, 1])
        assert got[col] == pytest.approx(r, abs=1e-6), col
    y = frame["stroke"].astype(float)
    groups = frame["work_type"]
    between = sum(len(g) * (g.mean() - y.mean()) ** 2 for _, g in y.groupby(groups))
    eta = np.sqrt(between / ((y - y.mean()) ** 2).sum())
    assert got["work_type"] == pytest.approx(eta, abs=1e-6)


def test_four_clones_give_the_same_ranking(stroke_csv, tmp_path):
    single = _single_pass(stroke_csv, tmp_path / "single")
    answer = _fleet(stroke_csv, tmp_path / "fleet", workers=4).answer
    assert _ranking(answer["metrics"]["drivers"]) == _ranking(single["metrics"]["drivers"])


def test_each_clone_is_cited_with_what_she_held(stroke_csv, tmp_path):
    answer = _fleet(stroke_csv, tmp_path / "fleet").answer
    texts = [c["finding"] for c in answer["contributions"]]
    assert len(texts) == 2 and "Ava-01 (rows 800–1,599): stroke in 0 of 800 rows" in texts[1], texts


def test_the_fixture_would_catch_per_clone_drivers(stroke_csv, tmp_path, monkeypatch):
    """The bite: rank from one clone's statistics (what a per-clone answer
    is) and the ranking is not the whole file's."""
    single = _single_pass(stroke_csv, tmp_path / "single")
    real = goal_merge.merge_driver_partials
    monkeypatch.setattr(goal_merge, "merge_driver_partials", lambda parts: real(parts[:1]))
    answer = _fleet(stroke_csv, tmp_path / "fleet").answer
    assert _ranking(answer["metrics"]["drivers"]) != _ranking(single["metrics"]["drivers"])


def test_an_identifier_too_wide_to_carry_is_not_ranked_by_either_path(tmp_path, monkeypatch):
    """diabetic_data's patient_nbr: 71,518 distinct of 101,766 rows, past
    the distinct values a partial carries. One pass and the fleet both read
    it as an identifier, as _is_identifier does, not as a driver."""
    import avaloka.goal_analysis as ga
    monkeypatch.setattr(ga, "_MAX_DISTINCT_CARRIED", 100)
    frame = _stroke_frame()
    rng = np.random.default_rng(3)
    frame["patient_nbr"] = rng.integers(0, int(len(frame) * 1.5), len(frame))
    path = tmp_path / "wide_id.csv"
    frame.to_csv(path, index=False)
    single = _single_pass(path, tmp_path / "single")
    answer = _fleet(path, tmp_path / "fleet").answer
    for drivers in (single["metrics"]["drivers"], answer["metrics"]["drivers"]):
        assert "patient_nbr" not in [d["column"] for d in drivers]
    assert _ranking(answer["metrics"]["drivers"]) == _ranking(single["metrics"]["drivers"])


# --------------------------------------------------------- (d) leakage fields
def _loan_frame(n: int = 1200, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grade = rng.choice(list("ABCDE"), n)
    rate = {"A": 7.0, "B": 10.0, "C": 14.0, "D": 18.0, "E": 23.0}
    int_rate = np.array([rate[g] for g in grade]) + rng.normal(0, 1.0, n)
    default = (rng.random(n) < (int_rate - 4) / 60).astype(int)
    amount = rng.integers(2, 40, n) * 1000.0
    return pd.DataFrame({
        "id": np.arange(n),
        "loan_amnt": amount,
        "int_rate": int_rate.round(2),
        "grade": grade,
        "annual_inc": rng.normal(70_000, 20_000, n).round(0),
        "initial_list_status": rng.choice(["w", "f"], n),
        # Recorded after the outcome: what was received, the last payment,
        # the settlement. Moderately associated, not >= 0.95.
        "total_rec_prncp": (amount * np.where(default == 1, rng.uniform(0.1, 0.6, n), 1.0)).round(2),
        "last_pymnt_amnt": (rng.normal(400, 120, n) * np.where(default == 1, 0.5, 1.0)).round(2),
        "settlement_amount": np.where(default == 1, rng.uniform(0, 5000, n), 0.0).round(2),
        "default": default,
    })


def _set_aside(result) -> str:
    return next((f for f in result["findings"] if f.startswith("Set aside as possible leakage")), "")


def test_post_outcome_fields_are_listed_as_possible_leakage_not_ranked():
    frame = _loan_frame()
    result = analyze_goal(frame, "what predicts default", {})
    assert result["kind"] == "drivers" and result["status"] == "answered", result
    ranked = [f.split(":")[0] for f in result["findings"][1:6]]
    note = _set_aside(result)
    for leak in ("total_rec_prncp", "last_pymnt_amnt", "settlement_amount"):
        assert leak not in ranked, ranked
        entry = next(d for d in result["metrics"]["drivers"] if d["column"] == leak)
        assert entry["leakage"] == "name"
        assert abs(entry["strength"]) < 0.95       # flagged by name, not by strength
        # Listed with its strength, never dropped silently.
        assert f"{leak} (" in note, note
    assert "name heuristic" in note and "Name one in the goal to rank it." in note
    assert "int_rate" in ranked or "grade" in ranked


def test_a_field_the_goal_names_is_ranked_even_if_it_looks_like_leakage():
    result = analyze_goal(_loan_frame(), "what predicts default with total_rec_prncp", {})
    ranked = [f.split(":")[0] for f in result["findings"][1:6]]
    assert "total_rec_prncp" in ranked, ranked
    assert "total_rec_prncp (" not in _set_aside(result)


def _visits_frame(n: int = 1500, seed: int = 4) -> pd.DataFrame:
    """Predictors whose names hold last/after/post/code: none is leakage."""
    rng = np.random.default_rng(seed)
    since = rng.integers(0, 365, n)
    calls = rng.poisson(2, n)
    postal = rng.choice([f"{z:05d}" for z in range(10_000, 10_020)], n)
    surname = rng.choice(["Ng", "Rao", "Kim", "Diaz", "Ali", "Cole", "Ito", "Lund", "Moss", "Park"], n)
    risky = {"10000", "10001", "10002", "10003"}
    logit = (-2.0 + 0.008 * since + 0.45 * calls + 1.4 * np.isin(postal, list(risky))
             + 1.2 * np.isin(surname, ["Ng", "Rao"]))
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"days_since_last_visit": since, "after_hours_calls": calls,
                         "postal_code": postal, "last_name": surname,
                         "noise": rng.normal(0, 1, n), "churned": churned})


BENIGN = ("postal_code", "last_name", "days_since_last_visit", "after_hours_calls")


def test_last_after_post_and_code_names_are_ranked_not_set_aside():
    result = analyze_goal(_visits_frame(), "what predicts churned", {})
    assert result["status"] == "answered", result
    ranked = [f.split(":")[0] for f in result["findings"][1:6]]
    for name in BENIGN:
        assert name in ranked, (name, ranked)
        entry = next(d for d in result["metrics"]["drivers"] if d["column"] == name)
        assert "leakage" not in entry
    assert _set_aside(result) == ""


def test_the_leakage_rule_matches_tokens_not_substrings():
    from avaloka.goal_analysis import likely_leakage
    for leak in ("debt_settlement_flag", "settlement_amount", "settlement_percentage", "total_rec_prncp",
                 "total_rec_late_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "recoveries",
                 "collection_recovery_fee", "hardship_flag", "orig_projected_additional_accrued_interest",
                 "last_fico_range_high", "lastPymntAmnt", "charge_off_date"):
        assert likely_leakage(leak) == "name", leak
    assert likely_leakage("final_payment_amount") == "timing"
    for fine in (*BENIGN, "post_code", "postcode", "final_grade", "days_after_discharge", "int_rate",
                 "age", "number_inpatient", "tip_amount", "pay_grade", "inq_last_6mths",
                 "mths_since_last_delinq", "chargeoff_within_12_mths"):
        assert likely_leakage(fine) is None, fine


def test_the_fixture_would_catch_a_substring_leakage_rule(monkeypatch):
    """The other direction: the first rule (#554 as raised) matched
    last/after/post/final as words alone, and these four were set aside."""
    import avaloka.goal_analysis as ga
    broad = re.compile(r"(settlement|^total_rec_|^last_pymnt|recoveries|post|code)", re.IGNORECASE)
    monkeypatch.setattr(ga, "likely_leakage", lambda name: (
        "name" if broad.search(name) else
        "timing" if ga._name_tokens(name) & {"last", "next", "final", "after", "post"} else None))
    result = analyze_goal(_visits_frame(), "what predicts churned", {})
    note = _set_aside(result)
    assert all(f"{name} (" in note for name in BENIGN), note


def test_the_fixture_would_catch_a_ranking_without_the_leakage_rule(monkeypatch):
    import avaloka.goal_analysis as ga
    monkeypatch.setattr(ga, "likely_leakage", lambda name: None)
    result = analyze_goal(_loan_frame(), "what predicts default", {})
    ranked = [f.split(":")[0] for f in result["findings"][1:6]]
    assert {"total_rec_prncp", "settlement_amount"} & set(ranked), ranked
