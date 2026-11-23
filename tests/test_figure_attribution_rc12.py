"""A figure given to an entity must be that entity's figure (rc12 web check).

work4 ("which work_type has the highest stroke rate?") was VERIFIED with
"Self‑employed: 0.079 stroke rate, 201 missing BMI records." 0.079 is
Self-employed's rate; 201 is the whole file's missing BMI (the code took
``df['bmi'].isna().sum()`` before grouping and put it beside the winner).
Self-employed has 44. The figure was in the result, so every check passed it.

The predictor threads (p1-p3, "what are the strongest predictors of
stroke?") showed figures beside columns that are not those columns'
association with stroke: in the data age r=0.245, heart disease 0.135,
glucose 0.132, hypertension 0.128, ever married eta=0.108. And p3's "Age is
the strongest predictor of stroke (importance ≈ 0.245)" was withheld though
0.245 is age's r: it is now shown relabelled.

Replies, code and result rows are the ones those runs produced
(results/rd_clean/work4, results/pred/p1-p3); no LLM runs. The work4 frame
reproduces the stroke file's group counts, strokes and missing BMI exactly;
the predictor ranking is the stroke file's own driver_ranking, pinned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.agents.claim_verifier as cv
from app.agents.claim_verifier import DataUsed, claim_verifier_node, presentable_reply, verify_claims, withheld_claims

# --------------------------------------------------------------------------- #
# work4
# --------------------------------------------------------------------------- #

WORK4_QUESTION = "which work_type has the highest stroke rate?"
WORK4_REPLY = ("Self‑employed workers have the highest stroke rate at **0.079 (≈7.9 %)**.  \n\n"
               "- Self‑employed: 0.079 stroke rate, 201 missing BMI records.")
WORK4_BULLET = "- Self‑employed: 0.079 stroke rate, 201 missing BMI records."
#: output_json as the API returned it: text cells, read back from the CSV.
WORK4_ROWS = [{"work_type_with_highest_stroke_rate": "Self-employed", "stroke_rate": "0.07936507936507936",
               "missing_bmi_count": "201"}]
WORK4_CODE = """import pandas as pd

def main(df):
    print('Columns and dtypes:')
    print(df.dtypes)
    missing_bmi = df['bmi'].isna().sum()
    df['stroke'] = pd.to_numeric(df['stroke'], errors='coerce')
    df_valid = df.dropna(subset=['work_type'])
    grouped = df_valid.groupby('work_type', as_index=False).agg(total_patients=('stroke', 'size'), stroke_cases=('stroke', 'sum'))
    grouped['stroke_rate'] = grouped['stroke_cases'] / grouped['total_patients']
    best_idx = grouped['stroke_rate'].idxmax()
    best_row = grouped.loc[best_idx]
    result = pd.DataFrame({'work_type_with_highest_stroke_rate': [best_row['work_type']], 'stroke_rate': [best_row['stroke_rate']], 'missing_bmi_count': [missing_bmi]})
    return result
"""
MISATTRIBUTED = "201 is the total missing BMI, not Self-employed's (44)"

#: The stroke file's work types: (rows, strokes, missing BMI).
WORK_TYPES = {"Private": (2925, 149, 114), "Self-employed": (819, 65, 44), "children": (687, 2, 16),
              "Govt_job": (657, 33, 27), "Never_worked": (22, 0, 0)}
#: Its genders: (rows, mean BMI).
GENDERS = {"Female": (2994, 29.065758), "Male": (2115, 28.647936), "Other": (1, 22.4)}


def work_frame() -> pd.DataFrame:
    """5,110 patients with the stroke file's per-work-type rows, strokes and
    missing BMI (201 in all), and its mean BMI per gender."""
    work, stroke, missing = [], [], []
    for name, (n, strokes, miss) in WORK_TYPES.items():
        work += [name] * n
        stroke += [1] * strokes + [0] * (n - strokes)
        missing += [True] * miss + [False] * (n - miss)
    n = len(work)
    gender = [g for g, (k, _b) in GENDERS.items() for _ in range(k)]
    rng = np.random.default_rng(7)
    gender = list(rng.permutation(gender))
    bmi = [np.nan if m else GENDERS[g][1] for m, g in zip(missing, gender)]
    return pd.DataFrame({"id": np.arange(67, 67 + n), "gender": gender, "age": rng.uniform(1, 82, n).round(0),
                         "hypertension": rng.integers(0, 2, n), "heart_disease": rng.integers(0, 2, n),
                         "ever_married": rng.choice(["Yes", "No"], n), "work_type": work,
                         "Residence_type": rng.choice(["Urban", "Rural"], n),
                         "avg_glucose_level": rng.uniform(55, 270, n).round(2), "bmi": bmi,
                         "smoking_status": rng.choice(["never smoked", "smokes", "Unknown"], n), "stroke": stroke})


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return work_frame()


@pytest.fixture(scope="module")
def used(frame) -> DataUsed:
    return DataUsed(frames=[("patients", frame)])


def test_the_frame_is_the_stroke_files_shape(frame):
    assert int(frame["bmi"].isna().sum()) == 201
    selfemp = frame[frame["work_type"] == "Self-employed"]
    assert int(selfemp["bmi"].isna().sum()) == 44
    assert round(selfemp["stroke"].mean(), 3) == 0.079


def _found(report, claim):
    return next((f for f in report.findings if f.claim == claim), None)


def test_work4_total_given_to_self_employed_is_withheld(used):
    report = verify_claims(WORK4_REPLY, {"execution_output": WORK4_ROWS}, used, question=WORK4_QUESTION,
                           code=WORK4_CODE)
    finding = _found(report, WORK4_BULLET)
    assert finding is not None, report.findings
    assert finding.check == "figure_misattributed"
    assert finding.detail == MISATTRIBUTED
    # 0.079 is Self-employed's: the headline sentence stands.
    assert _found(report, WORK4_REPLY.split("  \n")[0]) is None
    shown = presentable_reply(WORK4_REPLY, report.as_dict())
    assert "201 missing BMI" not in shown
    assert "highest stroke rate at **0.079" in shown
    assert f"Why: {MISATTRIBUTED}." in shown


def test_work4_is_not_verified_on_the_node_path(tmp_path, frame):
    """As the graph runs it: the frame the run read comes from execution_result.reads."""
    path = tmp_path / "patients.csv"
    frame.to_csv(path, index=False)
    state = {"messages": [HumanMessage(content=WORK4_QUESTION), AIMessage(content=WORK4_REPLY)],
             "coder_definition": {"code": WORK4_CODE},
             "execution_result": {"status": "success", "reads": [{"path": str(path), "used": True}]},
             "output_json": WORK4_ROWS}
    report = claim_verifier_node(state)["verification_report"]
    assert report["verified"] is False
    assert report["claims_reconciled"] is False
    assert {f["claim"]: f["detail"] for f in report["findings"]} == {WORK4_BULLET: MISATTRIBUTED}
    assert withheld_claims(report) == {WORK4_BULLET}


def test_another_groups_figure_is_withheld_with_whose_it_is(used):
    """Private's rate (0.051) given to Self-employed."""
    claim = "Self-employed workers have a stroke rate of 0.051."
    report = verify_claims(claim, {"execution_output": WORK4_ROWS}, used, question=WORK4_QUESTION)
    finding = _found(report, claim)
    assert finding is not None, report.findings
    assert finding.detail == "0.051 is Private's stroke rate, not Self-employed's (0.07937)"


def test_a_swapped_figure_in_the_result_table_is_withheld():
    """No frames: the result rows key the entities."""
    rows = [{"gender": "Female", "mean_bmi": 29.07}, {"gender": "Male", "mean_bmi": 28.65}]
    claim = "Mean BMI by gender: Female 28.65, Male 29.07."
    report = verify_claims(claim, {"execution_output": rows}, None, question="mean bmi by gender?")
    finding = _found(report, claim)
    assert finding is not None, report.findings
    assert finding.detail == "28.65 is Male's mean_bmi, not Female's (29.07)"


# --- guards: true claims stay ---------------------------------------------------

GUARDS = [
    ("Self-employed has the highest stroke rate (0.079).", WORK4_ROWS),
    ("201 patients are missing BMI.", WORK4_ROWS),
    ("Self-employed workers have the highest stroke rate, about 8%.", WORK4_ROWS),
    ("Self-employed: 44 missing BMI records out of 819 patients.",
     [{"work_type": "Self-employed", "patients": 819, "missing_bmi": 44}]),
    ("Self-employed workers make up 819 of the 5,110 patients.",
     [{"work_type": "Self-employed", "patients": 819, "all_patients": 5110}]),
    ("Female 29.07, Male 28.65.", [{"gender": "Female", "mean_bmi": 29.07}, {"gender": "Male", "mean_bmi": 28.65}]),
]


@pytest.mark.parametrize("claim,rows", GUARDS, ids=[g[0][:40] for g in GUARDS])
def test_true_figures_are_not_withheld(used, claim, rows):
    report = verify_claims(claim, {"execution_output": rows}, used, question=WORK4_QUESTION)
    assert not withheld_claims(report.as_dict()), report.findings


def test_tolerance_is_the_written_precision():
    from app.agents.claim_verifier import figure_matches

    assert figure_matches(0.079, "0.079", 0.0793650)
    assert not figure_matches(0.078, "0.078", 0.0793650)
    assert figure_matches(8, "8%", 0.0793650)                     # a percent of a fraction
    assert figure_matches(0.132, "0.132", 0.131945, approx=True)
    assert not figure_matches(0.129, "0.129", 0.134914, approx=True)
    assert figure_matches(0.13, "0.14", 0.127904, approx=True, high=0.14)    # "around 0.13–0.14"
    assert figure_matches(201, "201", 201.0) and not figure_matches(201, "201", 44.0)


# --------------------------------------------------------------------------- #
# Predictors: a figure beside a column is its association
# --------------------------------------------------------------------------- #

PRED_QUESTION = "what are the strongest predictors of stroke?"
#: driver_ranking on the stroke file (5,110 rows): what computed_drivers returns there.
RC12_RANKED = {
    "target": "stroke", "column": "stroke", "identifiers": ["id"],
    "drivers": [{"column": c, "measure": m, "strength": s} for c, m, s in [
        ("age", "pearson_r", 0.245257), ("heart_disease", "pearson_r", 0.134914),
        ("avg_glucose_level", "pearson_r", 0.131945), ("hypertension", "pearson_r", 0.127904),
        ("ever_married", "eta", 0.10834), ("work_type", "eta", 0.098087), ("smoking_status", "eta", 0.075525),
        ("bmi", "pearson_r", 0.042374), ("Residence_type", "eta", 0.015458), ("gender", "eta", 0.009617)]],
    "values": {"ever_married": ["Yes", "No"],
               "work_type": ["Private", "Self-employed", "children", "Govt_job", "Never_worked"],
               "Residence_type": ["Urban", "Rural"],
               "smoking_status": ["never smoked", "Unknown", "formerly smoked", "smokes"],
               "gender": ["Female", "Male", "Other"]},
}


@pytest.fixture
def ranked(monkeypatch):
    monkeypatch.setattr(cv, "computed_drivers", lambda question, used: RC12_RANKED)
    return RC12_RANKED


P1_REPLY = ("Age is the strongest predictor of stroke, with a correlation of 0.25.  \n"
            "- Heart disease, avg glucose level, and hypertension follow, each around 0.13–0.14.  \n"
            "- BMI, ever married, smoking status, and residence type have much smaller effects, the last barely 0.003.")
P1_ROWS = [{"predictor": p, "metric_type": t, "metric_value": v} for p, t, v in [
    ("age", "correlation", "0.24525734617097408"), ("heart_disease", "correlation", "0.13491399696869208"),
    ("avg_glucose_level", "correlation", "0.1319454408257098"), ("hypertension", "correlation", "0.12790382346647966"),
    ("gender", "max_rate_diff", "0.0487279843444227"), ("work_type", "max_rate_diff", "0.0487279843444227"),
    ("bmi", "correlation", "0.04237366114923351"), ("ever_married", "max_rate_diff", "0.03222257740076874"),
    ("smoking_status", "max_rate_diff", "0.03036806085331742"),
    ("Residence_type", "max_rate_diff", "0.0033819222919167363")]]
P1_CODE = """import pandas as pd

def main(df):
    rows = []
    overall_rate = df['stroke'].mean()
    for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
        group_rates = df[[col, 'stroke']].dropna().groupby(col)['stroke'].mean()
        rows.append({'predictor': col, 'metric_type': 'max_rate_diff',
                     'metric_value': float((group_rates - overall_rate).abs().max())})
    for col in ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease']:
        sub = df[[col, 'stroke']].dropna()
        rows.append({'predictor': col, 'metric_type': 'correlation', 'metric_value': float(abs(sub[col].corr(sub['stroke'])))})
    return pd.DataFrame(rows).sort_values('metric_value', ascending=False).reset_index(drop=True)
"""

P2_REPLY = "- **Next tier**: work_type_Self‑employed (0.32) and heart_disease (0.27) are also strong."
P2_ROWS = [{"predictor": p, "importance": v} for p, v in [
    ("work_type_children", "0.5082850100612494"), ("hypertension", "0.39711203748325813"),
    ("work_type_Self-employed", "0.3244587016701078"), ("heart_disease", "0.26960538372326476"),
    ("age", "0.07400026709209408")]]
P2_CODE = """import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def main(df):
    feature_cols = [c for c in df.columns if c not in ('id', 'stroke')]
    df_feat = pd.get_dummies(df[feature_cols].fillna(0), drop_first=False)
    model = LogisticRegression(max_iter=1000)
    model.fit(df_feat, df['stroke'].astype(int))
    coeffs = np.abs(model.coef_[0])
    return pd.DataFrame({'predictor': df_feat.columns.tolist(), 'importance': coeffs})
"""

P3_RELABEL = "Age is the strongest predictor of stroke (importance ≈ 0.245)."
P3_REPLY = (P3_RELABEL + "  \n"
            "- The next highest are avg_glucose_level (≈ 0.132) and heart_disease (≈ 0.129).  \n"
            "- Hypertension follows at ≈ 0.093, while BMI is much lower at ≈ 0.042.  \n"
            "- All other factors (marital status, work type, gender, smoking, residence) contribute less than 0.001.")
P3_ROWS = [{"predictor": p, "importance_score": v} for p, v in [
    ("age", "0.24525734617097467"), ("avg_glucose_level", "0.13194544082571014"),
    ("heart_disease", "0.12850251539517787"), ("hypertension", "0.0928510224767239"),
    ("bmi", "0.04237366114923359"), ("ever_married", "0.001205772150658193"),
    ("work_type", "0.0011736696237938978"), ("gender", "0.0008068558907220929"),
    ("smoking_status", "0.00040626662468321706"), ("Residence_type", "2.215795505456459e-05")]]
P3_CODE = """import numpy as np
import pandas as pd
from scipy import stats

def main(df):
    records = []
    for col in ['age', 'avg_glucose_level', 'bmi']:
        sub = df[['stroke', col]].dropna()
        corr, _ = stats.pointbiserialr(sub['stroke'], sub[col])
        records.append({'predictor': col, 'importance_score': abs(corr)})
    for col in ['hypertension', 'heart_disease']:
        rates = df.groupby(col)['stroke'].mean()
        records.append({'predictor': col, 'importance_score': abs(rates.iloc[0] - rates.iloc[1])})
    for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
        records.append({'predictor': col, 'importance_score': df.groupby(col)['stroke'].mean().var()})
    return pd.DataFrame(records).sort_values('importance_score', ascending=False).reset_index(drop=True)
"""


def _run(reply, rows, code):
    return verify_claims(reply, {"execution_output": rows}, DataUsed(), question=PRED_QUESTION, code=code)


def _lines(reply):
    return [line.strip() for line in reply.split("  \n")]


def test_p1_residence_types_figure_is_withheld_and_the_true_lines_stay(ranked):
    report = _run(P1_REPLY, P1_ROWS, P1_CODE)
    first, follow, last = _lines(P1_REPLY)
    assert withheld_claims(report.as_dict()) == {last}
    finding = _found(report, last)
    assert finding.check == "driver_figure_mismatch"
    assert finding.detail.startswith("it gives Residence_type 0.003, but Residence_type's association with "
                                     "stroke is eta=0.015")


def test_p2_next_tier_coefficients_are_withheld(ranked):
    report = _run(P2_REPLY, P2_ROWS, P2_CODE)
    finding = _found(report, P2_REPLY)
    assert finding is not None and finding.check == "driver_figure_mismatch"
    assert finding.detail.startswith("it gives work_type 0.32, but work_type's association with stroke is "
                                     "eta=0.098")
    assert "raw model coefficients" in finding.detail


@pytest.mark.parametrize("line,reason", [
    (1, "it gives heart_disease 0.129, but heart_disease's association with stroke is r=0.135"),
    (2, "it gives hypertension 0.093, but hypertension's association with stroke is r=0.128"),
    (3, "it says work_type, gender, smoking_status, Residence_type less than 0.001, but work_type's association "
        "with stroke is eta=0.098"),
])
def test_p3_figures_and_bounds_that_are_not_the_columns_are_withheld(ranked, line, reason):
    report = _run(P3_REPLY, P3_ROWS, P3_CODE)
    claim = _lines(P3_REPLY)[line]
    finding = _found(report, claim)
    assert finding is not None, [f.claim for f in report.findings]
    assert finding.detail.startswith(reason), finding.detail
    assert claim in withheld_claims(report.as_dict())


def test_p3_importance_equal_to_the_association_is_relabelled_not_withheld(ranked):
    report = _run(P3_REPLY, P3_ROWS, P3_CODE)
    finding = _found(report, P3_RELABEL)
    assert finding.check == "importance_relabelled"
    assert finding.rewrite == "Age is the strongest predictor of stroke (association r ≈ 0.245)."
    assert P3_RELABEL not in withheld_claims(report.as_dict())
    shown = presentable_reply(P3_REPLY, report.as_dict())
    assert shown.startswith("Age is the strongest predictor of stroke (association r ≈ 0.245).")
    assert "importance" not in shown.split("\n\n")[0]


def test_an_importance_that_is_not_the_association_is_still_withheld(ranked):
    claim = "Age is the strongest predictor of stroke (importance ≈ 0.199)."
    report = _run(claim, P3_ROWS, P3_CODE)
    assert _found(report, claim).check in {"importance_not_association", "driver_figure_mismatch"}
    assert withheld_claims(report.as_dict()) == {claim}


def test_followed_by_is_an_order(ranked):
    claim = "In the ranking, BMI is followed by age and heart disease."
    report = _run(claim, P1_ROWS, P1_CODE)
    finding = _found(report, claim)
    assert finding is not None, report.findings
    assert finding.detail.startswith("it puts bmi ahead of age")


def test_next_tier_does_not_hold_the_top_column(ranked):
    claim = "Next tier: age and hypertension are also strong."
    report = _run(claim, P1_ROWS, P1_CODE)
    finding = _found(report, claim)
    assert finding is not None, report.findings
    assert finding.detail.startswith("it puts age in the next tier")


@pytest.mark.parametrize("claim", [
    "Age is the strongest predictor (r = 0.245), followed by heart disease (0.135) and glucose (0.132).",
    "Gender and residence type each have associations below 0.05.",
    "Self-employed patients have a stroke rate of 0.079, the highest of any work type.",
])
def test_true_driver_figures_stay(ranked, claim):
    report = _run(claim, P1_ROWS + WORK4_ROWS, P1_CODE)
    assert not withheld_claims(report.as_dict()), report.findings
