"""The domain vocabulary: one planted dataset per domain, the generic fallback,
the privacy rule (names decide, values never leak) and the CLI wiring.

Mutation proof (run by hand, recorded in the PR): empty one domain's ``cues``
and its planted case below fails with that domain classified as something
else, so the per-domain assertions are load-bearing, not decorative.
"""

from __future__ import annotations

import json

import pytest

from avaloka.domains import (CONFIDENT, DOMAIN_IDS, GENERIC, VOCABULARY, classify, coerce,
                             hints_for, normalise)
from avaloka.mission.budget import Budget
from avaloka.mission.context import MissionContext, MissionKind
from avaloka.missions import run_analyze

# Column names only — no values, no customer content. Each set is the kind of
# header a public dataset in that domain actually ships with.
PLANTED = {
    "healthcare": ["patient_id", "admission_type", "diagnosis_code", "length_of_stay",
                   "total_charges", "readmitted"],
    "finance": ["account_number", "transaction_id", "amount", "currency", "balance",
                "merchant_name", "is_fraud"],
    "personal_finance": ["date", "payee", "category", "expense", "income", "savings", "budget_month"],
    "stock_market": ["Date", "Ticker", "Open", "High", "Low", "Adj Close", "Volume"],
    "retail": ["order_id", "sku", "product_name", "quantity", "unit_price", "discount", "customer_id"],
    "telecom": ["customerID", "tenure", "Contract", "MonthlyCharges", "TotalCharges",
                "InternetService", "Churn"],
    "genomics": ["gene_id", "chrom", "pos", "ref", "alt", "sample_id", "log2FoldChange", "padj"],
    "robotics": ["episode_index", "frame_index", "timestamp", "observation.state", "action",
                 "reward", "done"],
    "astronomy": ["objid", "ra", "dec", "u", "g", "r", "i", "z", "redshift", "class", "mjd"],
    "manufacturing": ["UDI", "Product ID", "Air temperature [K]", "Process temperature [K]",
                      "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Machine failure"],
    "hr": ["EmployeeNumber", "Department", "JobRole", "MonthlyIncome", "OverTime",
           "YearsAtCompany", "Attrition"],
    "education": ["student_id", "school", "study_time", "absences", "G1", "G2", "final_grade", "passed"],
    "logistics": ["shipment_id", "origin", "destination", "carrier", "weight_kg", "distance_km",
                  "delivery_date", "on_time"],
    "marketing": ["campaign_id", "channel", "impressions", "clicks", "conversions", "ad_spend", "roas"],
    "public_sector": ["fips", "county", "population", "median_income", "unemployment_rate",
                      "fiscal_year"],
}


def test_every_domain_the_plan_names_is_in_the_vocabulary():
    required = {"healthcare", "finance", "personal_finance", "stock_market", "retail", "telecom",
                "genomics", "robotics", "astronomy", "manufacturing", "hr", "education",
                "logistics", "marketing", "public_sector", GENERIC}
    assert required <= set(DOMAIN_IDS)
    assert set(PLANTED) == set(DOMAIN_IDS) - {GENERIC}


@pytest.mark.parametrize("domain_id", sorted(PLANTED))
def test_planted_dataset_is_read_as_its_domain(domain_id):
    result = classify(PLANTED[domain_id])
    assert result["domain"] == domain_id, result["candidates"]
    assert result["confidence"] >= CONFIDENT, "a planted set must not need the model"
    assert result["matched_cues"], "the match must say which columns decided it"
    assert {h["column"] for h in result["matched_cues"]} <= set(PLANTED[domain_id])


@pytest.mark.parametrize("domain_id", sorted(PLANTED))
def test_report_carries_one_sanity_check_of_that_domain(domain_id):
    """The first sanity check in the result is this domain's own, not a
    generic one and not another domain's."""
    result = classify(PLANTED[domain_id])
    checks = result["hints"]["sanity_checks"]
    assert checks and checks[0] in VOCABULARY[domain_id].sanity_checks
    for other in DOMAIN_IDS:
        if other != domain_id:
            assert checks[0] not in VOCABULARY[other].sanity_checks
    assert len(result["questions"]) >= 2
    assert result["questions"][0] in VOCABULARY[domain_id].questions


def test_every_domain_has_a_full_hint_set():
    for dom in VOCABULARY.values():
        assert 2 <= len(dom.sanity_checks) <= 4, dom.id
        assert 2 <= len(dom.questions) <= 3, dom.id
        assert dom.units and dom.targets and dom.join_keys, dom.id
        if dom.id != GENERIC:
            assert len(dom.cues) >= 8, dom.id
            assert dom.aliases, dom.id
        for q in dom.questions:
            assert q.endswith("?"), (dom.id, q)
    # Sanity checks are the per-domain proof in the plan; no two domains share one.
    all_checks = [c for d in VOCABULARY.values() for c in d.sanity_checks]
    assert len(all_checks) == len(set(all_checks))


def test_generic_fallback_states_its_confidence():
    result = classify(["a", "b", "c", "value_1", "value_2", "x"])
    assert result["domain"] == GENERIC
    assert result["confidence"] == 0.0
    assert "no column name matched" in result["reason"]
    assert result["hints"]["sanity_checks"] and result["questions"]
    assert result["candidates"] == []
    assert classify([]) ["domain"] == GENERIC


def test_a_tie_is_said_to_be_a_tie_and_names_no_domain():
    # One telecom cue and one marketing cue: neither should win outright.
    result = classify(["id", "churn", "campaign"])
    assert result["domain"] == GENERIC and result["confidence"] == 0.0
    assert "both" in result["reason"]
    assert result["tentative"]["domain"] == "telecom"
    assert result["tentative"]["confidence"] < CONFIDENT
    assert result["hints"] == VOCABULARY[GENERIC].hints()


def test_one_cue_is_not_enough_to_name_a_domain():
    """The Titanic passenger list has a ``Ticket`` column, which is one telecom
    cue. One cue names no domain: the read is generic, the candidate is kept
    as ``tentative`` (the app asks the model about it), and no telecom check
    reaches the report."""
    titanic = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch",
               "Ticket", "Fare", "Cabin", "Embarked"]
    result = classify(titanic)
    assert result["domain"] == GENERIC and result["confidence"] == 0.0
    assert result["tentative"] == {"domain": "telecom", "confidence": 0.57}
    assert "only 1 cue" in result["reason"]
    assert not set(result["hints"]["sanity_checks"]) & set(VOCABULARY["telecom"].sanity_checks)
    assert classify(PLANTED["telecom"])["tentative"] is None


def test_classification_is_deterministic_and_order_free():
    cols = PLANTED["healthcare"]
    a, b = classify(cols), classify(list(reversed(cols)))
    assert a["domain"] == b["domain"] and a["confidence"] == b["confidence"]
    assert a["hints"] == b["hints"] and a["questions"] == b["questions"]


def test_names_decide_and_sample_values_never_leak():
    """Privacy: ``sample_rows`` may confirm types (names grouped by type); no
    value, and nothing derived from one, reaches hints, questions or the
    rest of the result."""
    cols = PLANTED["healthcare"]
    secret = "ZQX-SENTINEL-9f3e"
    rows = [{"patient_id": secret, "total_charges": 1234.5, "length_of_stay": 3, "readmitted": "yes"},
            {"patient_id": secret + "b", "total_charges": 99.0, "length_of_stay": 1, "readmitted": "no"}]
    with_rows = classify(cols, sample_rows=rows)
    without = classify(cols)

    serialised = json.dumps(with_rows)
    assert secret not in serialised
    assert "1234.5" not in serialised and "yes" not in serialised
    assert with_rows["hints"] == without["hints"]
    assert with_rows["questions"] == without["questions"]
    assert with_rows["domain"] == without["domain"]
    assert with_rows["confidence"] == without["confidence"]
    # The only thing the rows contributed: which *matched* columns hold numbers /
    # text (total_charges matched no cue, so it is not even looked at).
    assert set(with_rows["type_check"]["numeric"]) == {"length_of_stay"}
    assert set(with_rows["type_check"]["text"]) == {"patient_id", "readmitted"}
    assert without["type_check"] == {"numeric": [], "text": []}

    # Hint text is published code: a value-shaped token has no business in it.
    for dom in VOCABULARY.values():
        for text in dom.sanity_checks + dom.questions + dom.units + dom.targets + dom.join_keys:
            assert "@" not in text and "http" not in text, (dom.id, text)


@pytest.mark.parametrize("free_text, expected", [
    ("FinTech/Payments", "finance"),
    ("Healthcare/Claims", "healthcare"),
    ("Scientific/Bioinformatics", "genomics"),
    ("Retail/E-commerce", "retail"),
    ("Marketing/Ads Attribution", "marketing"),
    ("Telecom churn", "telecom"),
    ("Stock Market", "stock_market"),
    ("hr", "hr"),
    ("Media/YouTube Analytics", GENERIC),
    # An alias matches at a word start only: "hr" is inside "Throughput",
    # "ads" inside "Downloads", "iot" inside "Patriot".
    ("Synchrotron Throughput", GENERIC),
    ("App Downloads", GENERIC),
    ("Patriot League Sports", GENERIC),
    ("HR analytics", "hr"),
    ("Unknown", GENERIC),
    ("", GENERIC),
    (None, GENERIC),
    (42, GENERIC),
])
def test_coerce_maps_free_text_onto_the_vocabulary_or_rejects_it(free_text, expected):
    assert coerce(free_text) == expected


def test_hints_for_unknown_id_is_generic():
    assert hints_for("no-such-domain")["domain"] == GENERIC
    assert hints_for("robotics")["questions"] == list(VOCABULARY["robotics"].questions)


def test_normalise_splits_camel_case_and_punctuation():
    assert normalise("PatientId") == "patient id"
    assert normalise("Adj Close") == "adj close"
    assert normalise("observation.state") == "observation state"
    assert normalise("Air temperature [K]") == "air temperature k"


def test_data_scout_profile_and_readme_carry_the_domain(churn_csv, tmp_path):
    """CLI wiring: ``data_quality.json`` has ``domain`` and the README names
    the domain with one of its sanity checks."""
    out = tmp_path / "ana"
    ctx = MissionContext(kind=MissionKind.ANALYZE, goal="Understand churn", data_source=str(churn_csv),
                         output_dir=out, budget=Budget(limit_usd=None))
    run_analyze(ctx)

    profile = json.loads((out / "data_quality.json").read_text())
    assert profile["domain"]["domain"] == "telecom"
    assert profile["domain"]["questions"] == list(VOCABULARY["telecom"].questions)

    readme = (out / "README.md").read_text()
    assert "## Domain" in readme
    assert "Telecom / churn" in readme
    assert any(check in readme for check in VOCABULARY["telecom"].sanity_checks)
    assert VOCABULARY["telecom"].questions[0] in readme


def _planted_csv(path, columns, n=40):
    """A CSV whose header is the planted column names and whose cells are
    integers: the domain must come from the names alone."""
    import csv
    import random

    rng = random.Random(7)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for _ in range(n):
            writer.writerow([rng.randint(0, 9) for _ in columns])
    return path


@pytest.mark.parametrize("domain_id", sorted(PLANTED))
def test_per_domain_the_report_names_the_domain_and_its_own_sanity_check(domain_id, tmp_path):
    """The plan's acceptance, end to end through the CLI: per domain, a planted
    dataset is profiled by the Data Scout (``data_quality.json`` carries the
    domain) and the Reporter's README names it with one of that domain's own
    sanity checks, not another domain's."""
    src = _planted_csv(tmp_path / "planted.csv", PLANTED[domain_id])
    out = tmp_path / "ana"
    ctx = MissionContext(kind=MissionKind.ANALYZE, goal="Describe the data", data_source=str(src),
                         output_dir=out, budget=Budget(limit_usd=None))
    run_analyze(ctx)

    profile = json.loads((out / "data_quality.json").read_text())
    assert profile["domain"]["domain"] == domain_id, profile["domain"]["candidates"]
    readme = (out / "README.md").read_text()
    section = readme.split("## Domain", 1)[1].split("\n## ", 1)[0]
    assert f"**{VOCABULARY[domain_id].label}**" in section
    own = [c for c in VOCABULARY[domain_id].sanity_checks if c in section]
    assert own, section
    for other in DOMAIN_IDS:
        if other != domain_id:
            assert not any(c in section for c in VOCABULARY[other].sanity_checks), other


def test_a_generic_dataset_gets_the_general_checks_in_the_report(tmp_path):
    src = _planted_csv(tmp_path / "opaque.csv", ["a", "b", "c", "value_1", "value_2"])
    out = tmp_path / "ana"
    ctx = MissionContext(kind=MissionKind.ANALYZE, goal="Describe the data", data_source=str(src),
                         output_dir=out, budget=Budget(limit_usd=None))
    run_analyze(ctx)
    readme = (out / "README.md").read_text()
    assert "No domain was recognised from the column names" in readme
    assert VOCABULARY[GENERIC].sanity_checks[0] in readme
    assert "read as" not in readme.split("## Deliverables", 1)[0].lower()
