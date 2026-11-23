"""The verifier's near-misses and false positives from the rc6 web re-check.

Every case is a reply stored by that run: a wrong statement that was shown, or
a correct one that was withheld. Replies, code and data are stubbed; no LLM
runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.claim_verifier import (DataUsed, claim_verifier_node, presentable_reply, verify_claims,
                                       withheld_claims)


def _stroke_frame(n: int = 4000) -> pd.DataFrame:
    """age has the strongest association (0.158); then hypertension 0.093,
    ever_married 0.083, heart_disease 0.071 (below 0.6 of age's),
    avg_glucose_level 0.062, work_type 0.011 (6th) and gender."""
    rng = np.random.default_rng(11)
    age = rng.uniform(1, 85, n)
    heart = (rng.random(n) < 0.08).astype(int)
    hyper = (rng.random(n) < 0.1).astype(int)
    glucose = rng.normal(100, 30, n).round(1)
    married = np.where(rng.random(n) < 0.6, "Yes", "No")
    work = rng.choice(["Private", "Self-employed", "children", "Govt_job"], n)
    p = (0.005 + (age / 85) ** 3 * 0.2 + heart * 0.08 + hyper * 0.07 + (glucose > 140) * 0.08
         + (married == "Yes") * 0.04)
    return pd.DataFrame({"age": age.round(1), "heart_disease": heart, "hypertension": hyper,
                         "avg_glucose_level": glucose, "ever_married": married, "work_type": work,
                         "gender": rng.choice(["Male", "Female"], n),
                         "stroke": (rng.random(n) < p).astype(int)})


STROKE = DataUsed(frames=[("stroke", _stroke_frame())])


def _driver_checks(reply: str, question: str = "What drives stroke?") -> dict:
    report = verify_claims(reply, {"execution_output": [{"feature": "x", "importance": 0.5}]}, STROKE,
                           question=question)
    return {f.claim: f.check for f in report.findings if f.check == "driver_ranking_contradicted"}


# --------------------------------------------------------------------------- #
# 1. "Heart disease is the strongest driver", with age named later
# --------------------------------------------------------------------------- #

def test_the_strongest_driver_is_judged_on_its_own_clause_not_on_every_column_named():
    """groq, rc6: heart_disease (0.135) is below 0.6 of age (0.245), but the
    claim also said "followed by older age", and naming age anywhere passed it."""
    claim = ("Heart disease is the strongest driver of stroke, with a stroke rate of 0.170, "
             "followed by older age (61-82 yrs, 0.139) and hypertension (0.133).")
    assert _driver_checks(claim) == {claim: "driver_ranking_contradicted"}
    right = "Age is the strongest driver of stroke, followed by hypertension and heart disease."
    assert _driver_checks(right) == {}


@pytest.mark.parametrize("phrase", [
    "Heart disease is the top factor for stroke.",
    "Heart disease is the leading predictor of stroke.",
    "The main driver of stroke is heart disease.",
    "**Primary factor**: heart disease.",
    "Heart disease is the **strongest** predictor of stroke.",
])
def test_every_strongest_phrasing_is_checked(phrase):
    assert _driver_checks(phrase) == {phrase: "driver_ranking_contradicted"}
    assert _driver_checks(phrase.replace("Heart disease", "Age").replace("heart disease", "age")) == {}


# --------------------------------------------------------------------------- #
# 2. "matters little" as a list, and a wrong ordinal rank
# --------------------------------------------------------------------------- #

def test_a_lower_impact_list_naming_the_top_driver_is_withheld():
    """groq, rc6: "**Lower-impact factors**: smoking status, age, ... all
    contribute less than 0.07" -- age has the strongest association."""
    claim = ("- **Lower‑impact factors**: gender, age, and glucose level all contribute less than 0.07.")
    assert _driver_checks(claim, "what predicts a stroke?") == {claim: "driver_ranking_contradicted"}
    fine = "- **Minor factors**: gender and glucose level contribute little."
    assert _driver_checks(fine, "what predicts a stroke?") == {}


def test_a_wrong_second_place_is_withheld_and_a_near_tie_is_not():
    """groq, rc6: "**Second**: work_type_children (0.50)"; work_type ranks 6th."""
    wrong = "- **Second**: work_type_children (≈ 0.50)."
    checks = _driver_checks(wrong, "what predicts a stroke?")
    assert checks == {wrong: "driver_ranking_contradicted"}
    report = verify_claims(wrong, {}, STROKE, question="what predicts a stroke?")
    assert "work_type ranks 6" in report.findings[0].detail
    # ever_married (3rd) and heart_disease (4th) are within fold noise of each other
    assert _driver_checks("- **Second**: hypertension.\n- **Third**: heart disease.",
                          "what predicts a stroke?") == {}
    assert _driver_checks("Heart disease is the second strongest predictor.", "what predicts a stroke?") == {}


# --------------------------------------------------------------------------- #
# 3. an empty join asserted anywhere, not only in the "how they relate" claim
# --------------------------------------------------------------------------- #

def _two_tables(tmp_path):
    customers, orders = tmp_path / "customers.csv", tmp_path / "orders.csv"
    pd.DataFrame({"customer_id": range(1, 51), "segment": ["smb", "consumer"] * 25}).to_csv(customers, index=False)
    pd.DataFrame({"order_id": range(100), "customer_id": [i % 50 + 1 for i in range(100)],
                  "amount": range(100)}).to_csv(orders, index=False)
    return [{"alias": "customers", "data_source_location": str(customers)},
            {"alias": "orders", "data_source_location": str(orders)}]


def test_a_next_step_saying_the_join_will_be_empty_is_withheld(tmp_path):
    """groq, rc6: "1. Show a preview of the join on `customer_id` (will be
    empty in this sample)." was shown; 2,000 ids appear in both tables."""
    item = "1. Show a preview of the join on `customer_id` (will be empty in this sample)."
    reply = ("You have two tables, customers and orders.\n\n**Next steps**\n" + item + "\n"
             "2. Show a preview of the join on `signup_year` (will produce matches).")
    out = claim_verifier_node({"messages": [AIMessage(content=reply)], "datasets_context": _two_tables(tmp_path)})
    report = out["verification_report"]
    assert [f["check"] for f in report["findings"]] == ["join_claim_contradicted"]
    shown = presentable_reply(reply, report)
    assert "will be empty" not in shown and "will produce matches" in shown


def test_a_join_that_might_be_empty_is_not_an_assertion(tmp_path):
    reply = "Check whether the join on customer_id is empty before you rely on it."
    out = claim_verifier_node({"messages": [AIMessage(content=reply)], "datasets_context": _two_tables(tmp_path)})
    assert out["verification_report"]["findings"] == []


# --------------------------------------------------------------------------- #
# 4. a group figure of a join is recomputed on that join
# --------------------------------------------------------------------------- #

MERGE_CODE = ("import pandas as pd\n"
              "def main(df_customers, df_orders):\n"
              "    merged = pd.merge(df_orders, df_customers, on='customer_id', how='left')\n"
              "    return merged.groupby('segment', as_index=False)['amount'].mean()\n")


def _segment_turn(tmp_path, reply: str, code: str = MERGE_CODE) -> dict:
    customers, orders = tmp_path / "customers.csv", tmp_path / "orders.csv"
    pd.DataFrame({"customer_id": [1, 2, 3, 4], "segment": ["enterprise", "smb", "consumer", "consumer"]}) \
        .to_csv(customers, index=False)
    pd.DataFrame({"order_id": range(8), "customer_id": [1, 1, 2, 2, 3, 3, 4, 4],
                  "amount": [500.0, 590.66, 180.0, 184.34, 50.0, 60.0, 64.58, 64.58]}).to_csv(orders, index=False)
    rows = [{"segment": "consumer", "amount": 59.79}, {"segment": "enterprise", "amount": 545.33},
            {"segment": "smb", "amount": 182.17}]
    return {"messages": [HumanMessage(content="How does the average order amount differ by customer segment?"),
                         AIMessage(content=reply)],
            "coder_definition": {"code": code}, "output_json": rows,
            "execution_result": {"status": "success", "reads": [
                {"path": str(customers), "alias": "customers"}, {"path": str(orders), "alias": "orders"}]}}


SEGMENT_REPLY = ("Enterprise customers spend the most on average, with $545.33 per order, followed by SMB at "
                 "$182.17, while consumer orders average only $59.79.\n"
                 "- Enterprise leads by a large margin over consumer (≈$485.54 difference).\n"
                 "- The gap between consumer and SMB is about $122.38.\n"
                 "- SMB customers have an average order amount of 182.17, which is higher than consumer orders.")


def test_segment_averages_of_the_codes_join_are_released(tmp_path):
    """groq and gemma, rc6: correct means of orders joined to customers were
    withheld as "group_figure_mismatch", recomputed on the unjoined frames."""
    report = claim_verifier_node(_segment_turn(tmp_path, SEGMENT_REPLY))["verification_report"]
    assert report["findings"] == [] and report["verified"] is True
    assert presentable_reply(SEGMENT_REPLY, report) == SEGMENT_REPLY


def test_a_wrong_segment_average_of_the_join_is_still_withheld(tmp_path):
    wrong = "Enterprise orders average $645.33."
    state = _segment_turn(tmp_path, wrong)
    state["output_json"] = [{"segment": "enterprise", "amount": 645.33}]
    report = claim_verifier_node(state)["verification_report"]
    assert [f["check"] for f in report["findings"]] == ["group_figure_mismatch"]
    assert "the join the code performed" in report["findings"][0]["detail"]


def test_a_join_that_cannot_be_reproduced_contradicts_nothing(tmp_path):
    index_join = MERGE_CODE.replace("pd.merge(df_orders, df_customers, on='customer_id', how='left')",
                                    "df_orders.join(df_customers.set_index('customer_id'), on='customer_id')")
    report = claim_verifier_node(_segment_turn(tmp_path, SEGMENT_REPLY, index_join))["verification_report"]
    assert not [f for f in report["findings"] if f["check"] == "group_figure_mismatch"]


# --------------------------------------------------------------------------- #
# 5. a training report quotes training_result
# --------------------------------------------------------------------------- #

TRAINING_RESULT = {"mlflow_run_id": "144a04959b2e4b4588056dcca007884b", "final_train_loss": 0.036973,
                   "final_val_loss": 0.044634, "final_accuracy": 0.963027, "final_f1_score": 0.952852,
                   "rows_processed": 5110}


def test_figures_quoted_from_training_result_are_shown():
    """gemma and groq, rc6: run id, losses and F1 copied from training_result
    were withheld as invented (8 statements)."""
    reply = ("* **MLflow Run ID:** 144a04959b2e4b4588056dcca007884b\n"
             "* **Final Train Loss:** 0.036973\n"
             "* **Final Validation Loss:** 0.044634\n"
             "* **Final F1 Score:** 0.952852")
    state = {"messages": [AIMessage(content=reply)], "training_result": TRAINING_RESULT,
             "evaluation_report": {"metrics": {"accuracy": 0.963027}, "beats_baseline": True}}
    report = claim_verifier_node(state)["verification_report"]
    assert not withheld_claims(report)
    invented = "* **Final Validation Loss:** 0.271828"
    state["messages"] = [AIMessage(content=reply + "\n" + invented)]
    assert withheld_claims(claim_verifier_node(state)["verification_report"]) == {invented}


# --------------------------------------------------------------------------- #
# 6. "never" in a category name is not absolute language
# --------------------------------------------------------------------------- #

def test_never_as_a_category_value_is_not_absolute_language():
    """groq, rc6: "never smoked (0.048)" and "never-worked (0.0)" name
    smoking_status and work_type values."""
    rows = [{"feature": "smoking_status", "category": "never smoked", "stroke_rate": 0.0476},
            {"feature": "work_type", "category": "Never_worked", "stroke_rate": 0.0},
            {"feature": "gender", "category": "Female", "stroke_rate": 0.0471}]
    reply = ("- **Lower-risk groups**: never smoked (0.048), female (0.047).\n"
             "- **Unexpected low rates**: never‑worked (0.0).\n"
             "- Patients who quit smoking never return to baseline risk.")
    report = verify_claims(reply, {"execution_output": rows})
    assert [(f.claim, f.check) for f in report.findings] == [
        ("- Patients who quit smoking never return to baseline risk.", "absolute_language")]


# --------------------------------------------------------------------------- #
# 7. ranks, differences and signs the matchers misread
# --------------------------------------------------------------------------- #

COUNTRIES = [{"Country": "United Kingdom", "Count": 19012}, {"Country": "Germany", "Count": 256},
             {"Country": "France", "Count": 197}]


def test_second_highest_is_a_rank_not_the_highest():
    """gemma, rc6: "Germany has the second-highest count at 256" was read as
    "highest" against the max 19,012."""
    right = "Germany has the second-highest count at 256, followed by France with 197."
    gap = "The gap between the top country and the second-place country is a difference of 18,756 entries."
    report = verify_claims(right + "\n" + gap + "\nThe United Kingdom has 19,012 rows.",
                           {"execution_output": COUNTRIES})
    assert report.findings == []
    wrong = verify_claims("France has the second-highest count at 197.", {"execution_output": COUNTRIES})
    assert [f.check for f in wrong.findings] == ["ordering_contradicted"]


def test_a_difference_of_two_stated_figures_is_supported_and_a_wrong_one_is_not():
    """groq, rc6: "about 9.3 percentage points" between 13.3% and 3.97% (rates
    0.1325 and 0.0397 in the result); gemma: "1,310 units fewer" where the two
    stated figures differ by 1,410, which stays withheld."""
    rates = [{"hypertension": 0, "stroke_rate": 0.039679}, {"hypertension": 1, "stroke_rate": 0.13253}]
    reply = ("Stroke rate is 13.3% with hypertension and 3.97% without.\n"
             "- Difference: about 9.3 percentage points.")
    assert verify_claims(reply, {"execution_output": rates}).findings == []
    sales = [{"item": "A", "units": 5147}, {"item": "B", "units": 3737}]
    wrong = ("The top item sold 5,147 units.\n"
             "The second highest seller sold 3,737 units, which is 1,310 units fewer than the top item.")
    report = verify_claims(wrong, {"execution_output": sales})
    assert [(f.check, "1,310" in f.detail) for f in report.findings] == [("invented_number", True)]
    assert verify_claims(wrong.replace("1,310", "1,410"), {"execution_output": sales}).findings == []


def test_a_non_breaking_hyphen_before_a_figure_is_a_minus():
    """groq, rc6: "congestion_surcharge (‑0.17)" against -0.170 in the result."""
    rows = [{"feature": "congestion_surcharge", "fare_correlation": -0.17000614699409147}]
    reply = "- **Weak or negative links**: congestion_surcharge (‑0.17) shows a small negative correlation."
    assert verify_claims(reply, {"execution_output": rows}).findings == []
    flipped = [{"feature": "congestion_surcharge", "fare_correlation": 0.17000614699409147}]
    assert [f.check for f in verify_claims(reply, {"execution_output": flipped}).findings] == ["invented_number"]
