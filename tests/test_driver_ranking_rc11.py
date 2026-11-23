"""rc11's "what are the strongest predictors of stroke?" threads (groq, clinic DB).

p3 was verified with 0 findings: "Age is the strongest predictor of stroke
(importance ~0.199), followed closely by average glucose level (~0.194) and the
patient ID variable (~0.189)." Its code fit a random forest through a
ColumnTransformer with ``remainder='passthrough'``, so ``id`` went into the
model, and reported ``feature_importances_`` (impurity, which favours a column
with many distinct values). p2 leaked "Hypertension and smoking_status are
close in rank, both far exceeding the next group (heart_disease ~0.27)", a
ranking with no "strongest" in it. In the data age ranks first (r=0.245), then
heart disease 0.135, glucose 0.132, hypertension 0.128; smoking is far below.

These pin the identifier rule, that a reply ranking an identifier is withheld,
that impurity scores are not reported as importance, that comparative rankings
are checked by #573's order rule, and what an omitted column means. Replies
and code are the ones those runs produced; no LLM runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.agents.claim_verifier import DataUsed, verify_claims, withheld_claims
from tests.test_driver_ranking_scale_free import QUESTION, stroke_frame

#: p3's reply as shown, and its first sentence.
P3_REPLY = ("Age is the strongest predictor of stroke (importance ≈ 0.199), followed closely by "
            "average glucose level (≈ 0.194) and the patient ID variable (≈ 0.189).  \n"
            "- **Top three**: age, avg_glucose_level, id.  \n"
            "- **Next tier**: bmi (≈ 0.168) and hypertension (≈ 0.025).  \n"
            "- **Low‑impact factors**: smoking status, work type, and gender all have importances below 0.02.  \n"
            "- **Very minor**: work_type_children, gender_Other, and work_type_Never_worked are essentially "
            "negligible.")
P3_CLAIM = P3_REPLY.split("  \n")[0]
#: p2's sentence that was shown (its other four were held back).
P2_CLAIM = ("- **Hypertension** and **smoking_status** are close in rank, both far exceeding the next group "
            "(heart_disease ≈ 0.27).")

#: p3's code (coder_definition.code, results/pred/p3/q0.json).
P3_CODE = """import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def main(df):
    df = df.dropna(subset=['stroke'])
    numeric_fill = ['age', 'avg_glucose_level', 'bmi']
    categorical_fill = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    for col in numeric_fill:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_fill:
        mode_val = df[col].mode(dropna=True)
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val.iloc[0])
        else:
            df[col] = df[col].fillna('missing')
    y = df['stroke']
    X = df.drop(columns=['stroke'])
    numeric_features = ['age', 'avg_glucose_level', 'bmi']
    categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_features), ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)], remainder='passthrough')
    model = Pipeline(steps=[('preprocess', preprocessor), ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    clf = model.named_steps['clf']
    importances = clf.feature_importances_
    num_names = numeric_features
    cat_encoder = model.named_steps['preprocess'].named_transformers_['cat']
    cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    passed_through = [c for c in X.columns if c not in numeric_features + categorical_features]
    feature_names = num_names + cat_names + passed_through
    importance_df = pd.DataFrame({'predictor': feature_names, 'importance': importances})
    importance_df = importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
    return importance_df
"""
#: Its result rows (the first eight of 22).
P3_ROWS = [{"predictor": p, "importance": v} for p, v in [
    ("age", 0.1987066246360252), ("avg_glucose_level", 0.19365810244873305), ("id", 0.18884627404666565),
    ("bmi", 0.16759259878244626), ("hypertension", 0.025415038999605327),
    ("heart_disease", 0.02160825582293308), ("gender_Female", 0.018645719839138885),
    ("smoking_status_never smoked", 0.018077083113485434)]]

#: The same model, with its importance measured by permutation on held-out rows.
PERMUTATION_CODE = P3_CODE.replace(
    "    importances = clf.feature_importances_\n",
    "    from sklearn.inspection import permutation_importance\n"
    "    importances = permutation_importance(model, X_val, y_val, random_state=0).importances_mean\n")

ID_REASON = "id is a row identifier, not a predictor"


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """The stroke frame with the Kaggle file's ``id``: unique integers 67..72940."""
    f = stroke_frame()
    f.insert(0, "id", np.random.default_rng(3).choice(np.arange(67, 72941), len(f), replace=False))
    return f


@pytest.fixture(scope="module")
def used(frame) -> DataUsed:
    return DataUsed(frames=[("patients", frame)])


def _checks(report) -> list:
    return [f.check for f in report.findings
            if f.check.startswith(("driver", "raw_", "identifier", "importance"))]


# --------------------------------------------------------------------------- #
# The identifier rule
# --------------------------------------------------------------------------- #

def _with_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Keys a name does not give away: pandas' saved index and a sparse
    auto-increment, each a little associated with the outcome by row order."""
    n = len(frame)
    rng = np.random.default_rng(5)
    f = frame.sort_values("age").reset_index(drop=True)
    f.insert(0, "Unnamed: 0", np.arange(n))                          # monotonic and dense
    f.insert(1, "seq", np.cumsum(rng.integers(1, 40, n)))            # monotonic, sparse
    f["patientId"] = rng.permutation(n) + 1000                       # *Id, dense
    f["income"] = rng.integers(10_000, 200_000, n)                   # near-unique, a measure
    return f


@pytest.mark.parametrize("shards", [1, 3])
def test_identifiers_are_never_ranked(frame, shards):
    from avaloka.goal_analysis import driver_partials, merge_driver_partials, outcome_expr, rank_drivers

    f = _with_keys(frame)
    parts = [driver_partials(part, "stroke", None, outcome_expr(f, "stroke"))
             for part in (f.iloc[i * len(f) // shards:(i + 1) * len(f) // shards] for i in range(shards))]
    drivers, skipped = rank_drivers(merge_driver_partials(parts))
    ranked = [d["column"] for d in drivers]
    for key in ("id", "Unnamed: 0", "seq", "patientId"):
        assert key not in ranked and key in skipped, key
    assert ranked[0] == "age"
    assert "income" in ranked                       # a near-unique measure spread wide is still a measure


@pytest.mark.parametrize("name, values, identifier", [
    ("id", np.random.default_rng(1).permutation(500) * 7, True),
    ("patient_id", np.arange(500), True),
    ("customerKey", np.arange(500).astype(str), True),
    ("uuid", [f"9f1c-{i:04d}" for i in range(500)], True),
    ("row", np.arange(500), True),
    ("Unnamed: 0", np.arange(500), True),
    ("seq", np.arange(500) * 13, True),                                     # monotonic
    ("record", np.random.default_rng(2).permutation(np.arange(100, 700))[:500], True),  # dense
    ("income", np.random.default_rng(4).integers(10_000, 2_000_000, 500), False),
    ("age", np.random.default_rng(4).integers(0, 90, 500), False),
    ("bmi", np.random.default_rng(4).normal(28, 6, 500).round(3), False),
    ("visit_counter", np.arange(500), True),                                # a row counter, any name
])
def test_the_identifier_rule(name, values, identifier):
    from avaloka.goal_analysis import is_identifier_column

    assert is_identifier_column(name, pd.Series(values)) is identifier


# --------------------------------------------------------------------------- #
# A reply that ranks an identifier is withheld
# --------------------------------------------------------------------------- #

def test_p3_ranking_the_patient_id_is_withheld(used):
    report = verify_claims(P3_CLAIM, {"execution_output": P3_ROWS}, used, question=QUESTION,
                           code=P3_CODE).as_dict()
    [finding] = report["findings"]
    assert finding["check"] == "identifier_as_predictor"
    assert finding["detail"].startswith(ID_REASON)
    assert "age has the strongest association with stroke (r=" in finding["detail"]
    assert "impurity scores (feature_importances_)" in finding["detail"]
    assert withheld_claims(report) == {P3_CLAIM}


def test_p3_reply_every_sentence_that_ranks_id_or_quotes_impurity_is_withheld(used):
    report = verify_claims(P3_REPLY, {"execution_output": P3_ROWS}, used, question=QUESTION,
                           code=P3_CODE).as_dict()
    by_claim = {f["claim"]: f["check"] for f in report["findings"]}
    assert by_claim[P3_CLAIM] == "identifier_as_predictor"
    assert by_claim["- **Top three**: age, avg_glucose_level, id."] == "identifier_as_predictor"
    assert by_claim["- **Low‑impact factors**: smoking status, work type, and gender all have importances "
                    "below 0.02."] == "importance_not_association"
    assert not report["safe_to_present"]


@pytest.mark.parametrize("claim", [
    "I left the id column out of the ranking because it is a row identifier.",
    "Age is the strongest predictor of stroke.",
])
def test_a_reply_that_sets_the_identifier_aside_passes(used, claim):
    assert _checks(verify_claims(claim, {}, used, question=QUESTION, code="")) == []


# --------------------------------------------------------------------------- #
# Importance wording
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("code, kind", [
    (P3_CODE, "impurity"),
    (PERMUTATION_CODE, "permutation"),
    ("def main(df):\n    return df.corr()\n", None),
], ids=["impurity", "permutation", "no_model"])
def test_what_importance_the_code_measured(code, kind):
    from app.agents.claim_verifier import model_importance

    assert model_importance(code) == kind


@pytest.mark.parametrize("code", [P3_CODE, "def main(df):\n    return df.corr()\n"], ids=["impurity", "no_model"])
def test_impurity_or_no_model_quoted_as_importance_is_withheld_even_in_the_right_order(used, code):
    claim = "Age is the strongest predictor of stroke (importance ≈ 0.199)."
    report = verify_claims(claim, {"execution_output": P3_ROWS}, used, question=QUESTION, code=code).as_dict()
    [finding] = report["findings"]
    assert finding["check"] == "importance_not_association"
    assert "ranked by association: age (r=" in finding["detail"]
    assert withheld_claims(report) == {claim}


def test_a_permutation_importance_in_the_right_order_stands(used):
    claim = "Age is the strongest predictor of stroke (permutation importance ≈ 0.031)."
    report = verify_claims(claim, {"execution_output": [{"predictor": "age", "importance": 0.031}]}, used,
                           question=QUESTION, code=PERMUTATION_CODE)
    assert _checks(report) == []


def test_a_permutation_importance_is_still_checked_against_the_association_ranking(used):
    claim = "Work type has the highest permutation importance (0.031)."
    report = verify_claims(claim, {"execution_output": [{"predictor": "work_type", "importance": 0.031}]}, used,
                           question=QUESTION, code=PERMUTATION_CODE)
    assert _checks(report) == ["driver_ranking_contradicted"]


# --------------------------------------------------------------------------- #
# Comparative rankings, with no "strongest"
# --------------------------------------------------------------------------- #



def test_p2_far_exceeding_is_withheld(used):
    report = verify_claims(P2_CLAIM, {"execution_output": []}, used, question=QUESTION, code="").as_dict()
    [finding] = report["findings"]
    assert finding["check"] == "driver_ranking_contradicted"
    assert finding["detail"].startswith("it puts smoking_status above heart_disease, but heart_disease ranks 2")
    assert withheld_claims(report) == {P2_CLAIM}


@pytest.mark.parametrize("claim, ahead, behind", [
    ("Smoking status ranks above age as a predictor of stroke.", "smoking_status", "age"),
    ("BMI is a stronger predictor of stroke than heart disease.", "bmi", "heart_disease"),
    ("Gender outweighs hypertension as a driver of stroke.", "gender", "hypertension"),
    ("Residence type and BMI are well above glucose in the ranking.", "Residence_type", "avg_glucose_level"),
])
def test_a_false_comparative_ranking_is_withheld(used, claim, ahead, behind):
    [finding] = verify_claims(claim, {}, used, question=QUESTION, code="").findings
    assert finding.check == "driver_ranking_contradicted"
    assert finding.detail.startswith(f"it puts {ahead} above {behind}")


@pytest.mark.parametrize("claim", [
    "age clearly exceeds heart_disease",
    "Age clearly exceeds heart_disease as a predictor of stroke.",
    "Age is a stronger predictor of stroke than BMI.",
    "Hypertension ranks above smoking status.",
    "Heart disease and glucose are close in rank, both well above BMI.",
    "Hypertension sits just above glucose in the ranking.",                 # a near-tie
    "Patients with hypertension have a stroke rate more than three times that of those without.",
    "Average glucose is higher among stroke patients than among others.",
])
def test_a_true_or_non_ranking_comparison_passes(used, claim):
    assert _checks(verify_claims(claim, {}, used, question=QUESTION, code="")) == []


# --------------------------------------------------------------------------- #
# Omissions and ordered chains
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("claim", [
    # p3's sentence without the identifier: heart disease (0.135) is left
    # out of a "followed by" list, but glucose (0.132) is within tolerance of
    # it, and a list with only an order is a selection, not a complete top 3.
    "Age is the strongest predictor of stroke, followed closely by average glucose level and hypertension.",
    "The three strongest predictors are age, glucose and hypertension.",
    "Top three: age, heart disease and glucose.",
])
def test_an_omission_within_tolerance_passes(used, claim):
    assert _checks(verify_claims(claim, {}, used, question=QUESTION, code="")) == []


def test_a_top_n_list_that_leaves_out_a_clearly_stronger_column_is_withheld(used):
    claim = "Top three: age, avg_glucose_level, bmi."
    [finding] = verify_claims(claim, {}, used, question=QUESTION, code="").findings
    assert finding.check == "driver_ranking_contradicted"
    assert finding.detail.startswith("it gives age, avg_glucose_level, bmi as the top 3, but leaves out heart_disease")


def test_a_followed_by_chain_after_a_singular_strongest_is_in_order(used):
    claim = "Age is the strongest predictor of stroke, followed by BMI and heart disease."
    [finding] = verify_claims(claim, {}, used, question=QUESTION, code="").findings
    assert finding.check == "driver_ranking_contradicted"
    assert finding.detail.startswith("it puts bmi ahead of heart_disease")


def test_the_coder_is_told_to_drop_identifiers_and_impurity_scores():
    import inspect

    from app.agents import coder

    source = inspect.getsource(coder)
    assert "Drop identifier columns before ranking or fitting anything" in source
    assert "Never use `remainder='passthrough'`" in source
    assert "NEVER report a tree model's `feature_importances_`" in source
