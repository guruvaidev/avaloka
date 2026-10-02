"""Can Avaloka actually do the everyday data-science job?

The rest of the suite asks whether the machinery runs. This asks whether it
gets the *analysis* right on the problems a data scientist meets in a normal
week: profile a messy export, pick the right task and the right split, refuse
to train on leaked data, tell an imbalanced problem from a balanced one, and
report what it found rather than that it finished.

Each test plants a fact and asserts Avaloka found it. The failure that matters
is not a crash -- it is a full, confident report that misses the leak, and from
the outside that looks exactly like success.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest


def _report(out, name):
    return json.loads((out / name).read_text())


def _checks(out):
    return {c["name"]: c for c in _report(out, "validation_report.json")["checks"]}


def _roles(out):
    return {c["name"]: c["role"] for c in _report(out, "data_quality.json")["columns"]}


def _issues(out):
    return {i["kind"] for i in _report(out, "data_quality.json")["quality"].get("issues", [])}


# ── Profiling: reading the data before modelling it ─────────────────────────

def test_column_roles_are_inferred_from_content(dirty, analyze):
    """Identifier, constant and categorical columns are told apart.

    Getting this wrong is not cosmetic: an identifier used as a feature
    memorises the training set, and a constant one wastes a slot.
    """
    _, out = analyze(dirty)
    roles = _roles(out)

    assert roles["row_id"] == "id"
    assert roles["country"] == "constant"
    assert roles["segment"] == "categorical"
    assert roles["amount"] == "numeric"


def test_a_messy_export_has_its_problems_named(dirty, analyze):
    """Missing values, duplicate rows and a constant column all surface."""
    _, out = analyze(dirty)
    issues = _issues(out)

    assert "duplicate_rows" in issues
    assert "constant_column" in issues
    assert "high_missingness" in issues, issues


def test_a_date_stored_as_text_is_recognised_as_a_date(time_ordered, analyze):
    """CSVs have no types, so a date arrives as a string.

    It used to be profiled as an identifier -- every value is unique -- which
    then denied the planner the one fact it needs to choose a temporal split.
    """
    _, out = analyze(time_ordered)
    assert _roles(out)["order_date"] == "datetime"


def test_an_identifier_is_not_mistaken_for_a_date(dirty, analyze):
    """The converse error is worse: it would order the split by an ID."""
    _, out = analyze(dirty)
    assert _roles(out)["row_id"] == "id"


# ── Planning: the right task, the right split ───────────────────────────────

def test_a_binary_target_gives_a_classification_plan(churn_like, train):
    _, out = train(churn_like, "churned", metric="roc_auc")
    plan = _report(out, "planner_graph.json")

    assert plan["task"] == "binary_classification"
    assert plan["split"]["strategy"] == "stratified_train_test", (
        "class balance must be preserved across the split"
    )


def test_a_continuous_target_gives_a_regression_plan(time_ordered, train):
    _, out = train(time_ordered, "revenue", metric="rmse")
    assert _report(out, "planner_graph.json")["task"] == "regression"


def test_time_ordered_data_is_split_by_time(time_ordered, train):
    """The classic silent error: random-splitting ordered data.

    Training on future rows to predict past ones flatters every metric, and
    nothing about the output looks wrong.
    """
    _, out = train(time_ordered, "revenue", metric="rmse")
    split = _report(out, "planner_graph.json")["split"]

    assert split["strategy"] == "time_based_holdout"
    assert "leak" in split["reason"].lower()


def test_the_planned_split_is_the_split_performed(time_ordered, train):
    """A reported split that did not happen is worse than a bad split.

    The planner chose a temporal split and the validator reported one while the
    model was fitted on a random shuffle. The claim has to be true.
    """
    from avaloka.modeling import _temporal_split

    frame = time_ordered
    features = frame[["promo", "base_demand"]]
    target = frame["revenue"]
    x_train, x_test, _, _ = _temporal_split(frame, features, target, "order_date", 0.2)

    train_dates = pd.to_datetime(frame.loc[x_train.index, "order_date"])
    test_dates = pd.to_datetime(frame.loc[x_test.index, "order_date"])

    assert train_dates.max() < test_dates.min(), (
        "every training row must precede every test row"
    )
    assert len(x_test) > 0 and len(x_train) > 0


def test_an_analysis_without_a_target_plans_no_split(churn_like, analyze):
    _, out = analyze(churn_like, goal="What is in this data?")
    plan = _report(out, "planner_graph.json")

    assert plan["task"] == "exploratory_analysis"
    assert plan["split"]["strategy"] == "none"


# ── Integrity: refusing to be fooled ────────────────────────────────────────

def test_a_leaked_feature_fails_the_mission(leaky, train):
    """collections_flag IS defaulted. A model scoring 1.0 has learned nothing."""
    _, out = train(leaky, "defaulted", metric="roc_auc")
    checks = _checks(out)

    assert checks["target_leakage"]["status"] == "fail"
    assert "collections_flag" in checks["target_leakage"]["detail"]
    assert _report(out, "validation_report.json")["verdict"] == "fail"


def test_a_leaked_model_is_never_certified_for_deployment(leaky, train):
    result, _ = train(leaky, "defaulted", metric="roc_auc")
    assert result.summary["max_safe_deployment_level"] == 1


def test_clean_data_is_not_accused_of_leaking(churn_like, train):
    """A detector that fires on everything is not a detector."""
    _, out = train(churn_like, "churned", metric="roc_auc")
    assert _checks(out)["target_leakage"]["status"] == "pass"


def test_a_constant_column_does_not_derail_the_mission(dirty, train):
    """Single-valued columns are ordinary, and used to crash the run.

    The engineer drops them before modelling while the leakage detector still
    walked the pre-transform profile, so it indexed a column that was no longer
    there and raised KeyError from inside the validator.
    """
    result, out = train(dirty, "converted", metric="roc_auc")
    assert result.summary["verdict"] in {"pass", "warn", "fail"}
    assert _checks(out)["target_leakage"]["status"] in {"pass", "fail"}


def test_duplicate_rows_are_removed_before_training(dirty, train):
    """Duplicates spanning a split inflate the score by memorisation."""
    _, out = train(dirty, "converted", metric="roc_auc")
    transformed = pd.read_parquet(out / "transformed_dataset.parquet")
    assert not transformed.duplicated().any(), "duplicates survived into training"


# ── Modelling: a number that means something ────────────────────────────────

def test_a_model_is_selected_and_scored(churn_like, train):
    result, _ = train(churn_like, "churned", metric="roc_auc")
    model = result.summary["model"]

    assert model["selected"]
    assert model["metric"] == "roc_auc"
    assert isinstance(model["score"], float)
    assert 0.0 <= model["score"] <= 1.0


def test_an_imbalanced_problem_is_not_scored_on_accuracy(imbalanced, train):
    """At 4% positives, always-negative scores 96% and is worthless."""
    result, out = train(imbalanced, "is_fraud", metric="roc_auc")

    assert result.summary["model"]["metric"] == "roc_auc"
    assert _report(out, "planner_graph.json")["split"]["strategy"] == "stratified_train_test"


def test_the_saved_model_actually_predicts(churn_like, train):
    """A packaged model that cannot be loaded and called is not a deliverable."""
    import pickle

    _, out = train(churn_like, "churned", metric="roc_auc", deployable=True)
    with (out / "model" / "model.pkl").open("rb") as fh:
        model = pickle.load(fh)

    frame = pd.read_parquet(out / "transformed_dataset.parquet")
    predictions = model.predict(frame.drop(columns=["churned"]).head(5))
    assert len(predictions) == 5


def test_cross_validation_reports_a_spread_not_just_a_mean(churn_like, train):
    """A single number hides how much the answer depends on the split."""
    _, out = train(churn_like, "churned", metric="roc_auc")
    checks = _checks(out)
    assert "stability" in checks
    assert any(token in checks["stability"]["detail"] for token in ("+/-", "std", "CV"))


# ── Reporting: saying what was found ────────────────────────────────────────

def test_an_analysis_states_findings_and_limitations(churn_like, analyze):
    result, out = analyze(churn_like, goal="Who churns and why?")
    readme = (out / "README.md").read_text()

    assert "## Key findings" in readme
    assert "## Limitations" in readme
    assert result.context.blackboard.get("findings"), "findings must reach the console too"


def test_small_data_is_declared_rather_than_glossed_over(analyze):
    """Twelve rows cannot support a conclusion, and the report must say so."""
    frame = pd.DataFrame({"x": range(12), "y": [0, 1] * 6})
    _, out = analyze(frame, goal="What can you tell me?")
    checks = _checks(out)

    assert checks["data_sufficiency"]["status"] == "warn"
    assert _report(out, "validation_report.json")["max_safe_deployment_level"] <= 2


def test_every_run_is_reproducible_from_its_own_bundle(churn_like, analyze):
    """Lineage, environment and the script that made the numbers."""
    _, out = analyze(churn_like)

    for artefact in ("analysis.py", "lineage.json", "environment.lock", "assumptions.yaml"):
        assert (out / artefact).exists(), f"missing {artefact}"

    lineage = _report(out, "lineage.json")
    assert lineage["source"]["sha256"], "the input must be pinned by hash"


def test_the_analysis_script_names_the_real_transformations(dirty, analyze):
    """The exported script has to reproduce the run, not describe it loosely."""
    _, out = analyze(dirty)
    script = (out / "analysis.py").read_text()

    assert "drop_duplicates" in script or "duplicate" in script
    assert "drop(columns=" in script


# ── The baseline: is the number better than doing nothing? ──────────────────

def test_a_real_signal_beats_the_trivial_baseline(churn_like, train):
    """Evidence means a comparison, not a score on its own."""
    result, out = train(churn_like, "churned", metric="roc_auc")
    checks = _checks(out)

    assert "beats_baseline" in checks, "every model must be compared to doing nothing"
    assert result.summary["model"]["baseline_strategy"] in {"prior", "most_frequent", "mean"}


def test_a_model_that_learned_nothing_is_reported_as_such(pure_noise, train):
    """The failure a baseline exists to catch.

    With no relationship in the data, a candidate still trains and still
    reports a score. Without a baseline that score reads as a result.
    """
    result, out = train(pure_noise, "outcome", metric="roc_auc")
    checks = _checks(out)

    assert checks["beats_baseline"]["status"] == "fail"
    assert result.summary["model"]["beats_baseline"] is False
    assert result.summary["max_safe_deployment_level"] == 1, (
        "a model that beats nothing is not deployable at any level above 1"
    )


def test_the_baseline_bar_is_the_interval_not_the_mean(pure_noise, train):
    """Beating the baseline by less than fold noise is not beating it.

    A majority-class classifier scores exactly the baseline, and floating-point
    noise then records it as a win by +0.0000.
    """
    from avaloka.modeling import train_candidates

    outcome = train_candidates(pure_noise, "outcome", "binary_classification",
                               metric="roc_auc")
    assert outcome.beats_baseline is False
    assert abs(outcome.baseline_score - 0.5) < 0.05, "prior baseline is 0.5 AUC"


def test_a_regression_baseline_is_the_training_mean(continuous_only, train):
    result, out = train(continuous_only, "throughput", metric="rmse")

    assert result.summary["model"]["baseline_strategy"] == "mean"
    assert _checks(out)["beats_baseline"]["status"] == "pass", (
        "throughput really is a function of latency and load"
    )


# ── Continuous features must survive to the model ───────────────────────────

def test_continuous_features_are_not_dropped_as_identifiers(continuous_only, train):
    """All-distinct is normal for a measurement, and is not an identifier.

    The rule dropped every float column whose values happened to be unique, so
    the model was fitted on whatever coarse columns remained -- or, when every
    feature was continuous, on nothing at all.
    """
    result, _ = train(continuous_only, "throughput", metric="rmse")

    assert result.summary["model"]["selected"]
    assert result.summary["model"]["score"] == result.summary["model"]["score"]  # not NaN


def test_an_identifier_column_is_still_kept_out_of_the_model(churn_like, train):
    """The rule still has to do the job it was there for."""
    from avaloka.modeling import _is_identifier_like

    assert _is_identifier_like(churn_like["customer_id"]) is True
    assert _is_identifier_like(churn_like["monthly_spend"]) is False
    assert _is_identifier_like(churn_like["tenure_months"]) is False
