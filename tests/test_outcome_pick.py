"""Which column the upload offers as the outcome, per dataset shape.

The web re-run asked "Is `VendorID` the outcome you want to understand?" of
the NYC taxi file (two vendor codes, so it looked two-valued) and offered
``initial_list_status`` (a loan's listing bookkeeping, "w" or "f") on the
LendingClub file, where the outcome is ``loan_status`` (nine values, so it
was never a candidate). An identifier-like name is never offered; a
bookkeeping field only when nothing stronger is there; a name that is the
outcome itself (loan_status, readmitted, default) may hold up to a dozen
values.

The fixtures are the column statistics the sampler reports for each file's
shape: names, types, distinct counts and the top value's share.
"""

from __future__ import annotations

from app.api import chat_voice


def _num(n_unique: int) -> dict:
    return {"type": "numeric", "n_unique": n_unique, "missing_ratio": 0.0}


def _cat(values: dict, total: int) -> dict:
    top = sorted(values.items(), key=lambda kv: -kv[1])
    return {"type": "categorical", "total_count": total, "missing_ratio": 0.0, "n_unique": len(values),
            "top_values": [{"value": k, "count": v, "frequency": v / total} for k, v in top]}


def _wide(n_unique: int, total: int) -> dict:
    return {"type": "categorical", "total_count": total, "missing_ratio": 0.0, "n_unique": n_unique,
            "top_values": [{"value": "x", "count": total // n_unique, "frequency": 1 / n_unique}]}


def _stats(columns: dict, rows: int) -> dict:
    return {"data_shape": {"rows": rows, "columns": len(columns)}, "column_statistics": columns}


def _taxi() -> dict:
    n = 200_000
    return _stats({
        "VendorID": _num(2), "tpep_pickup_datetime": _wide(190_000, n),
        "tpep_dropoff_datetime": _wide(190_000, n), "passenger_count": _num(9),
        "trip_distance": _num(3_000), "RatecodeID": _num(7),
        "store_and_fwd_flag": _cat({"N": 199_000, "Y": 1_000}, n), "PULocationID": _num(250),
        "DOLocationID": _num(259), "payment_type": _num(5), "fare_amount": _num(4_000),
        "extra": _num(40), "mta_tax": _num(5), "tip_amount": _num(3_000), "tolls_amount": _num(300),
        "improvement_surcharge": _num(4), "total_amount": _num(9_000),
        "congestion_surcharge": _num(4), "Airport_fee": _num(3),
    }, n)


def _loan() -> dict:
    n = 497_955
    return _stats({
        "id": {"type": "numeric", "n_unique": 0, "missing_ratio": 1.0},
        "loan_amnt": _num(1_500), "term": _cat({" 36 months": 350_000, " 60 months": 147_955}, n),
        "int_rate": _num(600), "grade": _cat({g: n // 7 for g in "ABCDEFG"}, n),
        "emp_length": _wide(11, n), "home_ownership": _cat({"MORTGAGE": 240_000, "RENT": 200_000,
                                                            "OWN": 57_000, "ANY": 955}, n),
        "annual_inc": _num(40_000),
        "verification_status": _cat({"Source Verified": 200_000, "Verified": 150_000,
                                     "Not Verified": 147_955}, n),
        "issue_d": _wide(130, n),
        "loan_status": _cat({"Fully Paid": 230_000, "Current": 150_000, "Charged Off": 90_000,
                             "Late (31-120 days)": 12_000, "In Grace Period": 8_000,
                             "Late (16-30 days)": 4_000, "Default": 1_000,
                             "Does not meet the credit policy. Status:Fully Paid": 2_000,
                             "Does not meet the credit policy. Status:Charged Off": 955}, n),
        "pymnt_plan": _cat({"n": 497_900, "y": 55}, n),
        "purpose": _wide(14, n),
        "initial_list_status": _cat({"w": 300_000, "f": 197_955}, n),
        "application_type": _cat({"Individual": 470_000, "Joint App": 27_955}, n),
        "total_rec_prncp": _num(200_000), "last_pymnt_d": _wide(130, n),
        "hardship_flag": _cat({"N": 497_000, "Y": 955}, n),
        "debt_settlement_flag": _cat({"N": 490_000, "Y": 7_955}, n),
    }, n)


def _diabetes() -> dict:
    n = 101_766
    return _stats({
        "encounter_id": _num(n), "patient_nbr": _num(71_518),
        "race": _cat({"Caucasian": 76_099, "AfricanAmerican": 19_210, "?": 2_273, "Hispanic": 2_037,
                      "Other": 1_506, "Asian": 641}, n),
        "gender": _cat({"Female": 54_708, "Male": 47_055, "Unknown/Invalid": 3}, n),
        "age": _cat({f"[{a}-{a + 10})": n // 10 for a in range(0, 100, 10)}, n),
        "admission_type_id": _num(8), "discharge_disposition_id": _num(26),
        "admission_source_id": _num(17), "time_in_hospital": _num(14),
        "medical_specialty": _wide(73, n), "num_lab_procedures": _num(118),
        "number_inpatient": _num(21), "diag_1": _wide(717, n),
        "max_glu_serum": _cat({"None": 96_420, "Norm": 2_597, ">200": 1_485, ">300": 1_264}, n),
        "A1Cresult": _cat({"None": 84_748, ">8": 8_216, "Norm": 4_990, ">7": 3_812}, n),
        "insulin": _cat({"No": 47_383, "Steady": 30_849, "Down": 12_218, "Up": 11_316}, n),
        "change": _cat({"No": 54_755, "Ch": 47_011}, n),
        "diabetesMed": _cat({"Yes": 78_363, "No": 23_403}, n),
        "readmitted": _cat({"NO": 54_864, ">30": 35_545, "<30": 11_357}, n),
    }, n)


def _question(stats: dict) -> str:
    return chat_voice.prompts_after_upload(sample_statistics=stats)[0]["question"]


def test_taxi_never_offers_an_identifier_like_code():
    read = chat_voice.read_statistics(_taxi())
    assert "VendorID" not in read.targets and "RatecodeID" not in read.targets, read.targets
    assert "VendorID" not in _question(_taxi())


def test_loan_offers_loan_status_not_initial_list_status():
    read = chat_voice.read_statistics(_loan())
    assert read.target == "loan_status", read.targets
    assert not {"initial_list_status", "pymnt_plan", "application_type", "hardship_flag",
                "debt_settlement_flag"} & set(read.targets), read.targets
    assert "`loan_status`" in _question(_loan())


def test_stroke_offers_stroke():
    from tests.test_honest_insights_narration import _stroke_stats
    read = chat_voice.read_statistics(_stroke_stats())
    assert read.target == "stroke", read.targets


def test_diabetes_offers_readmitted():
    read = chat_voice.read_statistics(_diabetes())
    assert read.target == "readmitted", read.targets
    assert not {"encounter_id", "patient_nbr"} & set(read.targets)


def test_a_bookkeeping_field_is_offered_when_nothing_stronger_is_there():
    only = _stats({"amount": _num(500), "initial_list_status": _cat({"w": 60, "f": 40}, 100)}, 100)
    assert chat_voice.read_statistics(only).target == "initial_list_status"


def test_the_alternative_offered_after_a_turn_is_not_bookkeeping():
    final = {"avaloka_intent": "statistical_analysis",
             "execution_result": {"status": "success", "message": "done", "output_row_count": 2},
             "output_json": [{"loan_status": "Charged Off", "n": 90_000}]}
    prompts = {p["step"]: p for p in chat_voice.prompts_after_turn(
        final, question="what drives loan_status?", sample_statistics=_loan())}
    replies = [r["label"] for r in prompts["understanding"]["quick_replies"]]
    assert "Use initial_list_status instead" not in replies, replies


def test_the_identifier_rule_is_case_sensitive_for_camel_case():
    for name in ("VendorID", "PULocationID", "PatientId", "provider", "zip_code", "patient_nbr"):
        assert chat_voice._ID_LIKE_RE.search(name), name
    for name in ("paid", "stroke", "Outcome", "readmitted", "loan_status", "valid"):
        assert not chat_voice._ID_LIKE_RE.search(name), name
