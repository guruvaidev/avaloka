""""What are the strongest predictors?" is answered by a scale-free ranking.

rc10 (clinic patients DB thread) answered "Highest importance: being married
(1.27) and living rurally (1.22) ... age (0.07)" from the absolute ``coef_`` of
a logistic regression on unscaled ``get_dummies`` features; rc7 (groq, stroke
upload) said "The strongest predictors of stroke are the work-type and smoking
variables". In the data age ranks first (r=0.245), then heart disease, glucose
and hypertension. These pin the one ranking every driver answer uses, the
verifier's check of strongest / top / key / highest-importance claims (singular
and plural) against it, and that a raw coefficient is never shown as importance.
Replies and code are the ones those runs produced; no LLM runs.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.claim_verifier import (DataUsed, claim_verifier_node, computed_drivers, presentable_reply,
                                       verify_claims, withheld_claims)

QUESTION = "what are the strongest predictors of stroke?"


def stroke_frame(n: int = 5110, seed: int = 11) -> pd.DataFrame:
    """The Kaggle stroke file's columns and associations (age r~0.23, heart
    disease, glucose and hypertension ~0.13, marriage eta ~0.11 through age),
    without shipping the file."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(0.1, 82, n).round(0)
    hypertension = (rng.random(n) < 0.02 + 0.25 * (age / 82) ** 2).astype(int)
    heart_disease = (rng.random(n) < 0.005 + 0.2 * (age / 82) ** 3).astype(int)
    glucose = (90 + 18 * rng.standard_normal(n) + 110 * (rng.random(n) < 0.08 + 0.12 * age / 82)).round(2)
    bmi = (24 + 6 * rng.standard_normal(n) + 6 * age / 82).round(1)
    married = np.where((age > 18) & (rng.random(n) < np.minimum(0.95, age / 45)), "Yes", "No")
    work = np.where(age < 16, "children", rng.choice(["Private", "Self-employed", "Govt_job", "Never_worked"], n,
                                                       p=[0.6, 0.2, 0.17, 0.03]))
    smoking = np.where(age < 16, rng.choice(["Unknown", "never smoked"], n, p=[0.8, 0.2]),
                       rng.choice(["never smoked", "formerly smoked", "smokes", "Unknown"], n,
                                  p=[0.4, 0.2, 0.17, 0.23]))
    logit = -7.4 + 0.065 * age + 0.6 * hypertension + 1.1 * heart_disease + 0.011 * (glucose - 90)
    stroke = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"gender": rng.choice(["Male", "Female"], n), "age": age, "hypertension": hypertension,
                         "heart_disease": heart_disease, "ever_married": married, "work_type": work,
                         "Residence_type": rng.choice(["Urban", "Rural"], n), "avg_glucose_level": glucose,
                         "bmi": bmi, "smoking_status": smoking, "stroke": stroke})


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return stroke_frame()


@pytest.fixture(scope="module")
def used(frame) -> DataUsed:
    return DataUsed(frames=[("patients", frame)])


#: The code rc10 ran (coder_definition.code, results/sweep/s2_pred/q0.json).
RC10_CODE = """import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def main(df):
    y = df['stroke']
    X_raw = df.drop(columns=['stroke'])
    numeric_fill_cols = ['age', 'avg_glucose_level', 'bmi']
    for col in numeric_fill_cols:
        median_val = X_raw[col].median()
        X_raw[col] = X_raw[col].fillna(median_val)
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    X_raw = X_raw.dropna(subset=categorical_cols)
    y = y.loc[X_raw.index]
    X_cat = pd.get_dummies(X_raw[categorical_cols].astype(str), drop_first=False)
    numeric_feature_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
    X_num = X_raw[numeric_feature_cols]
    X = pd.concat([X_num, X_cat], axis=1)
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X, y)
    coeff_abs = np.abs(model.coef_[0])
    importance_series = pd.Series(coeff_abs, index=X.columns)
    importance_df = importance_series.reset_index()
    importance_df.columns = ['feature', 'importance']
    importance_df = importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
    stroke_patients = df[df['stroke'] == 1]
    avg_age_stroke = stroke_patients['age'].mean()
    return importance_df
"""
#: Its result rows (the first eight and age, as output_json held them).
RC10_ROWS = [{"feature": f, "importance": v} for f, v in [
    ("ever_married_Yes", 1.2669758639585447), ("Residence_type_Rural", 1.2185101916371208),
    ("gender_Female", 1.1758188102643203), ("gender_Male", 1.159466215426157),
    ("Residence_type_Urban", 1.133592043398648), ("ever_married_No", 1.0851263710770047),
    ("work_type_Self-employed", 0.9205534990895079), ("smoking_status_never smoked", 0.7511066138558057),
    ("age", 0.06979197423603639), ("bmi", 0.008633901127040695), ("avg_glucose_level", 0.003937691245125214)]]
#: rc10's reply, as shown (before its own "held back" note).
RC10_REPLY = ("- **Highest importance**: being married (1.27) and living rurally (1.22).  \n"
              "- **Gender effect**: female (1.18) slightly outweighs male (1.16).  \n"
              "- **Other key factors**: unmarried status (1.09) and self‑employment (0.92).  \n"
              "- **Minor contributors**: age (0.07), BMI (0.01), and glucose level (0.004).")
#: rc7 item 1 (groq, stroke upload chat): the plural claim.
RC7_CLAIM = ("The strongest predictors of stroke are the work‑type and smoking variables, with age and "
             "glucose level also contributing.")


# --------------------------------------------------------------------------- #
# The one ranking
# --------------------------------------------------------------------------- #

def test_the_shared_ranking_puts_age_first_on_the_stroke_data(frame):
    from avaloka.goal_analysis import driver_ranking

    _stats, drivers, _skipped = driver_ranking(frame, "stroke")
    order = [d["column"] for d in drivers]
    assert order[0] == "age"
    assert set(order[1:4]) == {"heart_disease", "avg_glucose_level", "hypertension"}
    assert {d["column"]: d["measure"] for d in drivers}["ever_married"] == "eta"
    assert all(0 <= abs(d["strength"]) <= 1 for d in drivers)


@pytest.mark.skipif(not os.path.isfile(os.environ.get("AVALOKA_STROKE_CSV", "")),
                    reason="the Kaggle stroke file is not shipped; set AVALOKA_STROKE_CSV to it")
def test_the_shared_ranking_on_the_real_stroke_file():
    from avaloka.goal_analysis import driver_ranking

    real = pd.read_csv(os.environ["AVALOKA_STROKE_CSV"], na_values=["N/A"]).drop(columns="id")
    drivers = driver_ranking(real, "stroke")[1]
    assert [(d["column"], round(abs(d["strength"]), 3)) for d in drivers[:4]] == [
        ("age", 0.245), ("heart_disease", 0.135), ("avg_glucose_level", 0.132), ("hypertension", 0.128)]


def test_raw_coefficients_on_unscaled_dummies_put_a_dummy_above_age(frame):
    """Why the coefficient is not a ranking: the same data, rc10's model."""
    from sklearn.linear_model import LogisticRegression

    X = pd.get_dummies(frame.drop(columns="stroke"), dtype=float)
    model = LogisticRegression(max_iter=1000, solver="liblinear").fit(X, frame["stroke"])
    by_coef = pd.Series(np.abs(model.coef_[0]), index=X.columns).sort_values(ascending=False)
    assert by_coef.index[0] != "age" and by_coef["age"] < 0.1 < by_coef.iloc[0]


def test_the_verifier_ranks_with_the_shared_function(frame, used):
    from avaloka.goal_analysis import driver_ranking

    ranked = computed_drivers(QUESTION, used)
    assert [d["column"] for d in ranked["drivers"]] == [d["column"] for d in driver_ranking(frame, "stroke")[1]]
    assert "Rural" in ranked["values"]["Residence_type"]


# --------------------------------------------------------------------------- #
# The wrong claims are withheld
# --------------------------------------------------------------------------- #

def test_rc10_married_and_rural_above_age_is_withheld_on_the_node_path(tmp_path, frame):
    """The DB thread's reply, through claim_verifier_node as the graph runs it."""
    path = tmp_path / "patients.csv"
    frame.to_csv(path, index=False)
    state = {"messages": [HumanMessage(content=QUESTION), AIMessage(content=RC10_REPLY)],
             "coder_definition": {"code": RC10_CODE},
             "execution_result": {"status": "success", "reads": [{"path": str(path), "used": True}]},
             "output_json": RC10_ROWS}
    report = claim_verifier_node(state)["verification_report"]
    checks = {f["claim"]: f["check"] for f in report["findings"]}
    assert checks["- **Highest importance**: being married (1.27) and living rurally (1.22)."] == \
        "driver_ranking_contradicted"
    assert checks["- **Minor contributors**: age (0.07), BMI (0.01), and glucose level (0.004)."] == \
        "driver_ranking_contradicted"
    assert checks["- **Gender effect**: female (1.18) slightly outweighs male (1.16)."] == \
        "raw_coefficient_importance"
    assert checks["- **Other key factors**: unmarried status (1.09) and self‑employment (0.92)."] == \
        "raw_coefficient_importance"
    assert len(withheld_claims(report)) == 4
    shown = presentable_reply(RC10_REPLY, report)
    for figure in ("1.27", "1.22", "(0.07)", "1.18"):
        assert figure not in shown
    assert "age has the strongest association with stroke (r=0.2" in shown
    assert "raw model coefficients on unscaled features" in shown


def test_rc7_plural_strongest_predictors_claim_is_withheld(used):
    """"...are the work-type and smoking variables, with age ... also
    contributing": naming age after "with ... also" does not make it right."""
    report = verify_claims(RC7_CLAIM, {"execution_output": []}, used, question="what predicts a stroke?",
                           code="").as_dict()
    [finding] = report["findings"]
    assert finding["claim"] == RC7_CLAIM and finding["check"] == "driver_ranking_contradicted"
    assert "it ranks work_type among the strongest" in finding["detail"]
    assert "age has the strongest association with stroke (r=" in finding["detail"]
    assert presentable_reply(RC7_CLAIM, report).startswith("I've held back my answer")


@pytest.mark.parametrize("claim, ahead, behind", [
    ("Work type and smoking are the strongest predictors of stroke.", None, None),
    ("BMI and smoking are the strongest predictors, ahead of age.", None, None),
    ("Highest importance: marital status and residence.", None, None),
    ("The top predictors are age, then BMI, then heart disease.", "bmi", "heart_disease"),
    ("Key drivers: work type, followed by age.", None, None),
    ("Work type, then age, are the strongest predictors of stroke.", "work_type", "age"),
])
def test_a_claim_that_puts_a_weaker_column_first_is_withheld(used, claim, ahead, behind):
    report = verify_claims(claim, {}, used, question=QUESTION, code="")
    [finding] = report.findings
    assert finding.check == "driver_ranking_contradicted"
    if ahead:
        assert f"it puts {ahead} ahead of {behind}" in finding.detail
        assert "age has the strongest association with stroke (r=" in finding.detail


# --------------------------------------------------------------------------- #
# True claims pass
# --------------------------------------------------------------------------- #

TRUE_CLAIMS = [
    "Age is the strongest predictor of stroke.",
    "Age, then heart disease and glucose.",
    "The strongest predictors of stroke are age, then heart disease and glucose.",
    "Age, then heart disease and glucose, are the strongest predictors of stroke.",
    "The top predictors are age, heart disease, average glucose level and hypertension.",
    "Key drivers: age, heart disease and hypertension.",
    "Heart disease and age are the strongest predictors of stroke.",
    "The strongest predictors are heart disease, age and glucose.",
    "Age and glucose level are the top two predictors.",
    "Older age is the single biggest driver of stroke risk.",
    "Age is the most important predictor; heart disease and hypertension follow.",
    "Age matters most; other important factors include heart disease and glucose.",
    "Smoking is a key risk factor for stroke.",
    "The main drivers of stroke are age, hypertension and heart disease, while gender and residence matter little.",
    "Age is the strongest predictor of stroke (r=0.23), followed by heart disease.",
]


@pytest.mark.parametrize("claim", TRUE_CLAIMS)
def test_a_true_driver_claim_passes(used, claim):
    report = verify_claims(claim, {}, used, question=QUESTION, code="")
    assert [f.check for f in report.findings if f.check.startswith(("driver", "raw_"))] == []


@pytest.mark.parametrize("claim", TRUE_CLAIMS[:3])
def test_a_true_claim_with_no_coefficient_in_it_passes_after_raw_coefficient_code(used, claim):
    report = verify_claims(claim, {"execution_output": RC10_ROWS}, used, question=QUESTION, code=RC10_CODE)
    assert [f.check for f in report.findings if f.check.startswith(("driver", "raw_"))] == []


# --------------------------------------------------------------------------- #
# A raw coefficient is never importance
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("code, raw", [
    (RC10_CODE, True),
    ("from sklearn.linear_model import LinearRegression\n"
     "def main(df):\n    m = LinearRegression().fit(df[['a', 'b']], df['y'])\n    return m.coef_\n", True),
    (RC10_CODE.replace("    model = LogisticRegression",
                       "    from sklearn.preprocessing import StandardScaler\n"
                       "    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)\n"
                       "    model = LogisticRegression"), False),
    (RC10_CODE.replace("np.abs(model.coef_[0])", "np.abs(model.coef_[0] * X.std().to_numpy())"), False),
    ("from sklearn.ensemble import RandomForestClassifier\n"
     "def main(df):\n    m = RandomForestClassifier().fit(df[['a']], df['y'])\n    return m.feature_importances_\n",
     False),
    ("def main(df):\n    return df.corr()\n", False),
    ("def main(df) syntax error", False),
])
def test_raw_coefficient_importance_is_read_from_the_code(code, raw):
    from app.agents.claim_verifier import raw_coefficient_importance

    assert raw_coefficient_importance(code) is raw


def test_a_coefficient_quoted_as_importance_is_withheld_even_when_its_order_is_right(used):
    reply = "Age is the strongest predictor of stroke (importance 0.07)."
    report = verify_claims(reply, {"execution_output": RC10_ROWS}, used, question=QUESTION,
                           code=RC10_CODE).as_dict()
    [finding] = report["findings"]
    assert finding["check"] == "raw_coefficient_importance"
    assert "age has the strongest association with stroke" in finding["detail"]
    assert withheld_claims(report) == {reply}


def test_a_coefficient_read_as_a_per_unit_change_is_not_importance(used):
    reply = "Each extra year of age adds 0.07 to the log-odds of stroke."
    report = verify_claims(reply, {"execution_output": RC10_ROWS}, used, question=QUESTION, code=RC10_CODE)
    assert [f.check for f in report.findings if f.check.startswith(("driver", "raw_"))] == []


def test_the_same_figure_from_scaled_code_is_not_a_raw_coefficient(used):
    scaled = RC10_CODE.replace("    model = LogisticRegression",
                               "    from sklearn.preprocessing import StandardScaler\n"
                               "    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)\n"
                               "    model = LogisticRegression")
    reply = "Age is the strongest predictor of stroke (importance 0.07)."
    report = verify_claims(reply, {"execution_output": RC10_ROWS}, used, question=QUESTION, code=scaled)
    assert [f.check for f in report.findings if f.check.startswith(("driver", "raw_"))] == []


def test_the_coder_is_told_to_rank_scale_free():
    import inspect

    from app.agents import coder

    source = inspect.getsource(coder)
    assert "PREDICTOR / DRIVER RANKING LOGIC" in source
    assert "NEVER rank by a model's raw `coef_` on unscaled features" in source


# --------------------------------------------------------------------------- #
# The ranking is called association, never importance
# --------------------------------------------------------------------------- #

def test_the_cli_drivers_answer_calls_the_ranking_association(frame):
    from avaloka.goal_analysis import analyze_goal

    result = analyze_goal(frame, "what are the strongest predictors of stroke?", {})
    assert result["kind"] == "drivers" and result["status"] == "answered"
    text = " ".join(result["findings"])
    assert "age has the strongest association with stroke (r = 0.2" in text
    assert "does not control for the other columns" in text
    assert "importan" not in text.lower()


@pytest.mark.parametrize("claim, code", [
    (RC7_CLAIM, ""),                                                             # strongest, plural
    ("Work type, then age, are the strongest predictors of stroke.", ""),       # order
    ("Minor contributors: age (0.07), BMI (0.01).", ""),                         # weakest
    ("- **Gender effect**: female (1.18) slightly outweighs male (1.16).", RC10_CODE),  # raw coefficient
])
def test_every_withhold_reason_calls_the_ranking_association(used, claim, code):
    [finding] = [f for f in verify_claims(claim, {"execution_output": RC10_ROWS}, used, question=QUESTION,
                                          code=code).findings if f.check.startswith(("driver", "raw_"))]
    assert "association" in finding.detail
    assert "ranked by association: age (r=" in finding.detail
    assert "importan" not in finding.detail.lower()


def test_the_coder_is_told_to_call_it_association():
    import inspect

    from app.agents import coder

    source = inspect.getsource(coder)
    assert "Call the result association, never importance" in source
    assert "`association` sorted by `association` descending" in source
