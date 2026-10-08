"""wide-table rules for the Auto Insights prompt.

1. All columns in every call; example rows shrink with width, down to 1.
2. If still over the token budget: column parts, each with the key columns
   (IDs, outcome, date) and the same example record.
3. Parts' findings are consolidated; numbers are computed from the full rows.
"""
import json
import random
import threading

import pytest

import app.agents.visualization_agent as va


# ---------------------------------------------------------------- fixtures

def _icu_like_rows(n=2000, seed=7):
    """84 columns shaped like the ICU file: IDs, demographics, d1_/h1_ vitals,
    APACHE scores, comorbidity flags and a binary outcome."""
    rnd = random.Random(seed)
    vitals = ["diasbp", "heartrate", "mbp", "resprate", "spo2", "sysbp", "temp"]
    rows = []
    for i in range(n):
        r = {
            "encounter_id": 100000 + i * 3,
            "patient_id": 500000 + i * 7,
            "hospital_id": rnd.randint(1, 140),
            "age": rnd.randint(16, 89),
            "bmi": round(rnd.uniform(15, 50), 4),
            "elective_surgery": rnd.choice([0, 1]),
            "ethnicity": rnd.choice(["Caucasian", "African American", "Hispanic", "Asian", "Other"]),
            "gender": rnd.choice(["M", "F"]),
            "height": round(rnd.uniform(140, 200), 1),
            "icu_admit_source": rnd.choice(["Accident & Emergency", "Operating Room", "Floor", "Other Hospital"]),
            "icu_id": rnd.randint(1, 230),
            "icu_stay_type": rnd.choice(["admit", "transfer", "readmit"]),
            "icu_type": rnd.choice(["Med-Surg ICU", "MICU", "Neuro ICU", "CCU-CTICU", "SICU"]),
            "pre_icu_los_days": round(rnd.expovariate(1.5), 6),
            "weight": round(rnd.uniform(40, 150), 1),
        }
        for prefix in ("d1", "h1"):
            for v in vitals:
                base = rnd.uniform(40, 160)
                r[f"{prefix}_{v}_max"] = round(base + rnd.uniform(0, 30))
                r[f"{prefix}_{v}_min"] = round(base - rnd.uniform(0, 30))
        for prefix in ("d1", "h1"):
            for v in ("diasbp", "mbp", "sysbp"):
                r[f"{prefix}_{v}_noninvasive_max"] = r[f"{prefix}_{v}_max"]
                r[f"{prefix}_{v}_noninvasive_min"] = r[f"{prefix}_{v}_min"]
        for lab in ("glucose", "potassium"):
            r[f"d1_{lab}_max"] = round(rnd.uniform(3, 300), 2)
            r[f"d1_{lab}_min"] = round(rnd.uniform(3, 200), 2)
        for k in ("apache_2_diagnosis", "apache_3j_diagnosis", "heart_rate_apache",
                  "map_apache", "resprate_apache", "temp_apache", "gcs_eyes_apache",
                  "gcs_motor_apache", "gcs_verbal_apache"):
            r[k] = round(rnd.uniform(1, 300), 2)
        r["apache_4a_hospital_death_prob"] = round(rnd.random(), 3)
        r["apache_4a_icu_death_prob"] = round(rnd.random(), 3)
        for f in ("aids", "cirrhosis", "diabetes_mellitus", "hepatic_failure",
                  "immunosuppression", "leukemia", "lymphoma"):
            r[f] = rnd.choice([0, 0, 0, 1])
        r["apache_3j_bodysystem"] = rnd.choice(["Cardiovascular", "Respiratory", "Neurological", "Sepsis", "Metabolic"])
        r["apache_2_bodysystem"] = rnd.choice(["Cardiovascular", "Respiratory", "Neurologic", "Metabolic"])
        r["hospital_death"] = rnd.choice([0] * 11 + [1])
        rows.append(r)
    return rows


@pytest.fixture(scope="module")
def icu_rows():
    rows = _icu_like_rows()
    assert len(rows[0]) >= 80
    return rows


@pytest.fixture(scope="module")
def narrow_rows(icu_rows):
    keep = list(icu_rows[0].keys())[:15]
    return [{k: r[k] for k in keep} for r in icu_rows]


def _profile(rows):
    cols = va.profile_columns(rows)
    return {
        "dataset": {"dataset_id": "t", "rows_sampled": len(rows), "columns": [c["name"] for c in cols]},
        "columns": cols,
        "feature_ranking": va.compute_unsupervised_feature_ranking(cols),
    }


# ---------------------------------------------------------------- compact JSON

def test_prompt_numbers_are_rounded_not_truncated_in_meaning():
    out = json.loads(va._prompt_json({"mean": 25497.327055738206, "ratio": 0.00014285714285714287,
                                      "skew": 0.7237161184961719, "n": 120, "bad": float("nan")}))
    assert out == {"mean": 25497.33, "ratio": 0.0001429, "skew": 0.7237, "n": 120, "bad": None}


def test_prompt_json_has_no_indentation():
    assert "\n" not in va._prompt_json({"a": [1, 2, {"b": 3}]})


# ---------------------------------------------------------------- key columns

def test_key_columns_are_ids_and_outcome(icu_rows):
    keys = va._detect_key_columns(va.profile_columns(icu_rows))
    assert keys[:2] == ["encounter_id", "patient_id"]
    assert "hospital_death" in keys
    # A probability column is not an outcome: it is not binary.
    assert "apache_4a_hospital_death_prob" not in keys
    assert len(keys) <= va._MAX_KEY_COLUMNS


def test_named_target_is_the_first_key(icu_rows):
    keys = va._detect_key_columns(va.profile_columns(icu_rows), target_column="hospital_death")
    assert keys[0] == "hospital_death"


# ---------------------------------------------------------------- rule 1

def test_narrow_table_is_one_call_with_example_rows(narrow_rows, monkeypatch):
    monkeypatch.setattr(va, "_PROMPT_TOKEN_BUDGET", 6000)
    plans, info = va._plan_insight_requests(_profile(narrow_rows), narrow_rows)
    assert info["mode"] == "single" and len(plans) == 1
    assert 1 <= info["example_rows"] <= va._MAX_EXAMPLE_ROWS
    assert info["est_tokens"][0] <= 6000
    assert len(plans[0]["columns"]) == 15


def test_rows_shrink_to_one_before_any_split(narrow_rows, monkeypatch):
    prof = _profile(narrow_rows)
    monkeypatch.setattr(va, "_PROMPT_TOKEN_BUDGET", 100000)
    _, roomy = va._plan_insight_requests(prof, narrow_rows)
    with_one = va._estimate_tokens(va._insight_prompt(
        {**prof, "example_rows": va._example_rows(narrow_rows, [0], prof["dataset"]["columns"])}, None))
    monkeypatch.setattr(va, "_PROMPT_TOKEN_BUDGET", with_one + 5)
    plans, tight = va._plan_insight_requests(prof, narrow_rows)
    assert roomy["example_rows"] == va._MAX_EXAMPLE_ROWS
    assert tight["mode"] == "single" and tight["example_rows"] == 1
    assert len(plans[0]["columns"]) == 15          # still every column


# ---------------------------------------------------------------- rule 2

def test_wide_table_splits_with_keys_and_same_record_in_every_part(icu_rows, monkeypatch):
    monkeypatch.setattr(va, "_PROMPT_TOKEN_BUDGET", 6000)
    prof = _profile(icu_rows)
    plans, info = va._plan_insight_requests(prof, icu_rows)
    assert info["mode"] == "split" and len(plans) >= 2
    keys = info["key_columns"]
    all_cols = prof["dataset"]["columns"]
    seen = []
    example_ids = set()
    for p in plans:
        names = [c["name"] for c in p["columns"]]
        assert names[:len(keys)] == keys                    # keys in every part
        assert p["dataset"]["columns"] == all_cols          # full column list for context
        assert len(p["example_rows"]) == 1
        example_ids.add(p["example_rows"][0]["encounter_id"])
        seen += p["split"]["columns_in_part"]
    assert len(example_ids) == 1                            # the same record everywhere
    non_keys = [c for c in all_cols if c not in keys]
    assert seen == non_keys                                 # each column exactly once, in order
    assert all(t <= 6000 for t in info["est_tokens"])


def test_parts_are_capped(icu_rows, monkeypatch):
    monkeypatch.setattr(va, "_PROMPT_TOKEN_BUDGET", 1500)
    monkeypatch.setattr(va, "_MAX_SPLITS", 3)
    plans, info = va._plan_insight_requests(_profile(icu_rows), icu_rows)
    assert info["splits"] == 3 and "capped" in info["note"]


# ---------------------------------------------------------------- rule 3

def test_consolidation_takes_each_parts_best_first_and_drops_repeats():
    a = [{"type": "bar", "x_field": "icu_type"}, {"type": "histogram", "x_field": "age"}]
    b = [{"type": "histogram", "x_field": "d1_heartrate_max"}, {"type": "bar", "x_field": "icu_type"}]
    merged = va._consolidate_split_suggestions([a, b])
    assert [m["x_field"] for m in merged] == ["icu_type", "d1_heartrate_max", "age"]


def _fake_llm_factory(prompts, fail_part=None):
    lock = threading.Lock()

    class Resp:
        def __init__(self, content):
            self.content = content

    def fake(prompt, effort=None, *a, **k):
        with lock:
            prompts.append(prompt)
        payload = json.loads(prompt.split("Dataset profile JSON:\n", 1)[1].rsplit("\n\nReturn ONLY", 1)[0])
        part = (payload.get("split") or {}).get("part")
        if fail_part and part == fail_part:
            raise RuntimeError("boom")
        cols = [c for c in payload["columns"]
                if c["role"] != "identifier" and c["dtype"] == "numeric"
                and c["name"] not in (payload.get("split") or {}).get("key_columns", [])]
        out = [{"insight": f"{c['name']} varies.", "type": "histogram", "title": c["name"],
                "intent": "distribution", "x_field": c["name"], "y_field": None} for c in cols[:3]]
        return Resp(json.dumps(out))
    return fake


def test_wide_build_runs_parts_in_parallel_and_grounds_on_full_rows(icu_rows, monkeypatch):
    monkeypatch.setattr(va, "_PROMPT_TOKEN_BUDGET", 6000)
    monkeypatch.setattr(va, "viz_llm", object())
    prompts = []
    monkeypatch.setattr(va, "invoke_viz_llm", _fake_llm_factory(prompts))
    cfg = va.build_visualization_config_from_sample("t", icu_rows)
    plan = cfg["insight_model"]["prompt_plan"]
    assert plan["mode"] == "split" and len(prompts) == plan["splits"]
    for p in prompts:
        assert "hospital_death" in p and "encounter_id" in p
    assert cfg["insight_model"]["used"] is True
    llm_charts = [c for c in cfg["charts"] if c["id"].startswith("llm_")]
    assert len(llm_charts) >= 3
    # Charts come from different parts, and their numbers from ALL rows.
    assert len({c["encodings"]["x"]["field"][:3] for c in llm_charts}) >= 2
    for c in llm_charts:
        assert c["derived_data"]["n"] == len(icu_rows)


def test_one_failing_part_does_not_lose_the_others(icu_rows, monkeypatch):
    monkeypatch.setattr(va, "_PROMPT_TOKEN_BUDGET", 6000)
    monkeypatch.setattr(va, "viz_llm", object())
    prompts = []
    monkeypatch.setattr(va, "invoke_viz_llm", _fake_llm_factory(prompts, fail_part=1))
    cfg = va.build_visualization_config_from_sample("t", icu_rows)
    assert cfg["insight_model"]["used"] is True
    assert cfg["insight_model"]["error"] is None


def test_single_path_still_goes_through_llm_suggest_chart_specs(narrow_rows, monkeypatch):
    """Existing tests patch _llm_suggest_chart_specs; the narrow path must keep calling it."""
    monkeypatch.setattr(va, "viz_llm", object())
    calls = []
    def fake(viz_profile, user_context=None, effort=None):
        calls.append(viz_profile)
        return [{"insight": "x", "type": "histogram", "title": "Age", "x_field": "age"}], None
    monkeypatch.setattr(va, "_llm_suggest_chart_specs", fake)
    cfg = va.build_visualization_config_from_sample("t", narrow_rows)
    assert len(calls) == 1 and "example_rows" in calls[0]
    assert cfg["insight_model"]["prompt_plan"]["mode"] == "single"


def test_parts_are_balanced_not_one_tiny_leftover(icu_rows, monkeypatch):
    monkeypatch.setattr(va, "_PROMPT_TOKEN_BUDGET", 6000)
    _, info = va._plan_insight_requests(_profile(icu_rows), icu_rows)
    sizes = info["columns_per_part"]
    assert len(sizes) >= 2
    assert min(sizes) >= 0.5 * max(sizes), sizes






"""Auto Insights fixes: duplicate findings (#2), 0/1 outcomes (#3),
text vs chart numbers (#5), full-file charts (#4)."""
import csv

import numpy as np

import app.agents.visualization_agent as va

SYSTEMS = (["Cardiovascular"] * 34 + ["Neurological"] * 13 + ["Sepsis"] * 13
           + ["Respiratory"] * 12 + ["Gastrointestinal"] * 10 + ["Trauma"] * 16 + [""] * 2)


def _rows(n=400):
    out = []
    for i in range(n):
        age = 20 + (i * 13) % 70                      # 20..89
        death = 1 if (i * 37) % 100 < (age - 20) else 0   # rate rises with age
        out.append({"encounter_id": 1000 + i, "age": age, "hospital_death": death,
                    "apache_3j_bodysystem": SYSTEMS[i % len(SYSTEMS)]})
    return out


def _profiles(rows):
    ps = va.profile_columns(rows)
    return ps, {p["name"]: p for p in ps}


def _llm_charts(suggestions, rows, ps):
    return va._charts_from_llm_suggestions(
        suggestions, {"dataset": {"rows_sampled": len(rows)}, "columns": ps})


# ---- #2 duplicates ----
def test_same_columns_with_another_chart_type_is_one_finding():
    merged = va._consolidate_split_suggestions(
        [[{"type": "pie", "x_field": "hospital_death"}],
         [{"type": "bar", "x_field": "hospital_death"}]],
        key_columns=["encounter_id", "hospital_death"])
    assert len(merged) == 1


def test_only_one_key_only_finding_is_kept():
    merged = va._consolidate_split_suggestions(
        [[{"type": "pie", "x_field": "hospital_death"}, {"type": "histogram", "x_field": "age"}],
         [{"type": "bar", "x_field": "hospital_death", "y_field": "encounter_id"},
          {"type": "bar", "x_field": "apache_3j_bodysystem"}]],
        key_columns=["encounter_id", "hospital_death"])
    assert [m["x_field"] for m in merged] == ["hospital_death", "age", "apache_3j_bodysystem"]


# ---- #3 0/1 outcome ----
def test_scatter_of_binary_outcome_becomes_rate_by_band():
    rows = _rows()
    ps, by = _profiles(rows)
    charts = _llm_charts([{"type": "scatter", "x_field": "age", "y_field": "hospital_death",
                           "title": "Age vs Hospital Death",
                           "insight": "Older patients show higher death counts."}], rows, ps)
    assert charts[0]["type"] == "bar" and charts[0]["config"]["bin_x"]
    assert charts[0]["insight"] is None
    kept = va._ground_all(charts, rows, by)
    pts = kept[0]["derived_data"]["points"]
    assert 4 <= len(pts) <= 8
    assert all(0 <= p["y"] <= 100 for p in pts)
    assert pts[-1]["y"] > pts[0]["y"]
    assert "rate" in kept[0]["insight"] and kept[0]["insight_source"] == "data"


def test_scatter_with_outcome_on_x_is_swapped():
    rows = _rows()
    ps, _ = _profiles(rows)
    charts = _llm_charts([{"type": "scatter", "x_field": "hospital_death", "y_field": "age",
                           "insight": "x"}], rows, ps)
    assert charts[0]["encodings"]["x"]["field"] == "age"
    assert charts[0]["encodings"]["y"]["field"] == "hospital_death"


# ---- #5 text vs chart ----
def test_llm_share_that_disagrees_with_the_chart_is_replaced():
    rows = _rows()
    ps, by = _profiles(rows)
    charts = _llm_charts([{"type": "pie", "x_field": "apache_3j_bodysystem", "title": "Body systems",
                           "insight": "Cardiovascular dominates (33.5% of records)."}], rows, ps)
    ch = va._ground_all(charts, rows, by)[0]
    assert ch["insight_source"] == "data"
    assert ch["derived_data"]["llm_insight_replaced"].startswith("Cardiovascular")
    assert "34.7%" in ch["insight"]        # 136 of the 392 rows that have a value


def test_llm_share_that_matches_the_chart_is_kept():
    rows = _rows()
    ps, by = _profiles(rows)
    charts = _llm_charts([{"type": "pie", "x_field": "apache_3j_bodysystem",
                           "insight": "Cardiovascular is the largest group (34.7%)."}], rows, ps)
    assert va._ground_all(charts, rows, by)[0]["insight_source"] == "llm"


# ---- #4 full file ----
def test_charts_are_computed_on_the_full_file(tmp_path, monkeypatch):
    monkeypatch.setattr(va, "viz_llm", None)
    rows = _rows(2000)
    path = tmp_path / "full.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cfg = va.build_visualization_config_from_sample(
        "ds1", rows[:300], full_data_path=str(path), full_data_type="csv", total_rows=2000)
    assert cfg["dataset"]["rows_charted"] == 2000
    assert cfg["dataset"]["charts_scope"] == "full_file"
    texts = [c.get("insight") or "" for c in cfg["charts"]]
    assert any("full file" in t for t in texts)
    assert not any("in the sample" in t for t in texts)


def test_missing_full_file_keeps_the_sample_charts(monkeypatch):
    monkeypatch.setattr(va, "viz_llm", None)
    rows = _rows(300)
    cfg = va.build_visualization_config_from_sample("ds2", rows, full_data_path="/nonexistent.csv")
    assert cfg["dataset"]["charts_scope"] == "sample"


def test_to_float_accepts_numpy_numbers():
    assert va._to_float(np.int64(7)) == 7.0
    assert va._to_float(np.float32(1.5)) == 1.5
    assert va._to_float(np.float64("nan")) is None




def test_histogram_points_are_keyed_for_the_frontend():
    rows = _rows()
    ps, by = _profiles(rows)
    charts = _llm_charts([{"type": "histogram", "x_field": "age", "insight": None}], rows, ps)
    pts = va._ground_all(charts, rows, by)[0]["derived_data"]["points"]
    assert all("age" in p and "count" in p for p in pts)
    assert sum(p["count"] for p in pts) == len(rows)


def test_outcome_rate_chart_is_added_when_missing(monkeypatch):
    monkeypatch.setattr(va, "viz_llm", None)
    rows = _rows(1000)
    cfg = va.build_visualization_config_from_sample("ds3", rows)
    rate = [c for c in cfg["charts"]
            if ((c.get("encodings") or {}).get("y") or {}).get("field") == "hospital_death"]
    assert rate, "expected a hospital_death rate chart"
    assert rate[0]["encodings"]["x"]["field"] == "age"      # widest rate spread in the test data
    assert "rate" in rate[0]["insight"]


def test_llm_bar_of_outcome_by_category_is_a_rate():
    rows = _rows()
    ps, by = _profiles(rows)
    charts = _llm_charts([{"type": "bar", "x_field": "apache_3j_bodysystem",
                           "y_field": "hospital_death", "insight": "x"}], rows, ps)
    ch = va._ground_all(charts, rows, by)[0]
    assert ch["config"]["rate_by_group"] and ch["insight"].startswith("hospital_death rate by")
    assert all(0 <= p["y"] <= 100 for p in ch["derived_data"]["points"])


    """Auto Insights fixes: duplicate findings, 0/1 outcomes, text vs chart numbers,
full-file charts, histogram points, outcome-rate chart, number parsing."""
import csv

import numpy as np

import app.agents.visualization_agent as va

SYSTEMS = (["Cardiovascular"] * 34 + ["Neurological"] * 13 + ["Sepsis"] * 13
           + ["Respiratory"] * 12 + ["Gastrointestinal"] * 10 + ["Trauma"] * 16 + [""] * 2)


def _rows(n=400):
    out = []
    for i in range(n):
        age = 20 + (i * 13) % 70                          # 20..89
        death = 1 if (i * 37) % 100 < (age - 20) else 0   # rate rises with age
        out.append({"encounter_id": 1000 + i, "age": age, "hospital_death": death,
                    "apache_3j_bodysystem": SYSTEMS[i % len(SYSTEMS)]})
    return out


def _profiles(rows):
    ps = va.profile_columns(rows)
    return ps, {p["name"]: p for p in ps}


def _llm_charts(suggestions, rows, ps):
    return va._charts_from_llm_suggestions(
        suggestions, {"dataset": {"rows_sampled": len(rows)}, "columns": ps})


# ---- duplicates ----
def test_same_columns_with_another_chart_type_is_one_finding():
    merged = va._consolidate_split_suggestions(
        [[{"type": "pie", "x_field": "hospital_death"}],
         [{"type": "bar", "x_field": "hospital_death"}]],
        key_columns=["encounter_id", "hospital_death"])
    assert len(merged) == 1


def test_only_one_key_only_finding_is_kept():
    merged = va._consolidate_split_suggestions(
        [[{"type": "pie", "x_field": "hospital_death"}, {"type": "histogram", "x_field": "age"}],
         [{"type": "bar", "x_field": "hospital_death", "y_field": "encounter_id"},
          {"type": "bar", "x_field": "apache_3j_bodysystem"}]],
        key_columns=["encounter_id", "hospital_death"])
    assert [m["x_field"] for m in merged] == ["hospital_death", "age", "apache_3j_bodysystem"]


# ---- 0/1 outcome ----
def test_scatter_of_binary_outcome_becomes_rate_by_band():
    rows = _rows()
    ps, by = _profiles(rows)
    charts = _llm_charts([{"type": "scatter", "x_field": "age", "y_field": "hospital_death",
                           "title": "Age vs Hospital Death",
                           "insight": "Older patients show higher death counts."}], rows, ps)
    assert charts[0]["type"] == "bar" and charts[0]["config"]["bin_x"]
    assert charts[0]["insight"] is None
    kept = va._ground_all(charts, rows, by)
    pts = kept[0]["derived_data"]["points"]
    assert 4 <= len(pts) <= 8
    assert all(0 <= p["y"] <= 100 for p in pts)
    assert pts[-1]["y"] > pts[0]["y"]
    assert "rate" in kept[0]["insight"] and kept[0]["insight_source"] == "data"


def test_scatter_with_outcome_on_x_is_swapped():
    rows = _rows()
    ps, _ = _profiles(rows)
    charts = _llm_charts([{"type": "scatter", "x_field": "hospital_death", "y_field": "age",
                           "insight": "x"}], rows, ps)
    assert charts[0]["encodings"]["x"]["field"] == "age"
    assert charts[0]["encodings"]["y"]["field"] == "hospital_death"


def test_binary_column_by_itself_is_described_as_a_rate():
    rows = _rows()
    ps, by = _profiles(rows)
    charts = _llm_charts([{"type": "bar", "x_field": "hospital_death", "insight": None}], rows, ps)
    ch = va._ground_all(charts, rows, by)[0]
    assert ch["insight"].startswith("hospital_death is 1 for ")
    assert {p["x"] for p in ch["derived_data"]["points"]} == {"hospital_death = 0", "hospital_death = 1"}


def test_llm_bar_of_outcome_by_category_is_a_rate():
    rows = _rows()
    ps, by = _profiles(rows)
    charts = _llm_charts([{"type": "bar", "x_field": "apache_3j_bodysystem",
                           "y_field": "hospital_death", "insight": "x"}], rows, ps)
    ch = va._ground_all(charts, rows, by)[0]
    assert ch["config"]["rate_by_group"] and ch["insight"].startswith("hospital_death rate by")
    assert all(0 <= p["y"] <= 100 for p in ch["derived_data"]["points"])


# ---- text vs chart ----
def test_llm_share_that_disagrees_with_the_chart_is_replaced():
    rows = _rows()
    ps, by = _profiles(rows)
    charts = _llm_charts([{"type": "pie", "x_field": "apache_3j_bodysystem", "title": "Body systems",
                           "insight": "Cardiovascular dominates (33.5% of records)."}], rows, ps)
    ch = va._ground_all(charts, rows, by)[0]
    assert ch["insight_source"] == "data"
    assert ch["derived_data"]["llm_insight_replaced"].startswith("Cardiovascular")
    assert "34.7%" in ch["insight"]
    assert "rows with a value for apache_3j_bodysystem" in ch["insight"]


def test_llm_share_that_matches_the_chart_is_kept():
    rows = _rows()
    ps, by = _profiles(rows)
    charts = _llm_charts([{"type": "pie", "x_field": "apache_3j_bodysystem",
                           "insight": "Cardiovascular is the largest group (34.7%)."}], rows, ps)
    assert va._ground_all(charts, rows, by)[0]["insight_source"] == "llm"


def test_typographic_minus_and_rounding_are_accepted():
    meta = {"n": 1000, "min": 16, "max": 89, "median": 65, "mean": 62.31, "skewness": -0.6512}
    assert va._llm_text_conflict("Left-skewed (skewness \u22120.65), median 65.", "histogram", meta, False) is None
    meta2 = {"n": 1000, "min": 90, "max": 232, "median": 146, "mean": 148.34, "skewness": 0.58}
    assert va._llm_text_conflict("Slightly right-skewed (skew 0.6), mean 148.3.", "histogram", meta2, False) is None


def test_wrong_numbers_are_still_caught():
    meta = {"n": 1000, "min": 16, "max": 89, "median": 65, "mean": 62.31, "skewness": -0.65}
    assert va._llm_text_conflict("Median age is 71.", "histogram", meta, False)
    assert va._llm_text_conflict("Skewness is +0.65.", "histogram", meta, False)


# ---- histogram points ----
def test_histogram_points_are_keyed_for_the_frontend():
    rows = _rows()
    ps, by = _profiles(rows)
    charts = _llm_charts([{"type": "histogram", "x_field": "age", "insight": None}], rows, ps)
    pts = va._ground_all(charts, rows, by)[0]["derived_data"]["points"]
    assert all("age" in p and "count" in p for p in pts)
    assert sum(p["count"] for p in pts) == len(rows)


# ---- outcome-rate chart ----
def test_outcome_rate_chart_is_added_when_missing(monkeypatch):
    monkeypatch.setattr(va, "viz_llm", None)
    rows = _rows(1000)
    cfg = va.build_visualization_config_from_sample("ds3", rows)
    rate = [c for c in cfg["charts"]
            if ((c.get("encodings") or {}).get("y") or {}).get("field") == "hospital_death"]
    assert rate, "expected a hospital_death rate chart"
    assert rate[0]["encodings"]["x"]["field"] == "age"
    assert "rate" in rate[0]["insight"]


def test_prediction_columns_are_not_used_as_rate_drivers():
    rows = _rows(1000)
    for r in rows:
        r["apache_4a_hospital_death_prob"] = 0.9 if r["hospital_death"] else 0.1
    ps, _ = _profiles(rows)
    best = va._best_rate_driver(rows, ps, "hospital_death")
    assert best and best[1] == "age"


# ---- full file ----
def test_charts_are_computed_on_the_full_file(tmp_path, monkeypatch):
    monkeypatch.setattr(va, "viz_llm", None)
    rows = _rows(2000)
    path = tmp_path / "full.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cfg = va.build_visualization_config_from_sample(
        "ds1", rows[:300], full_data_path=str(path), full_data_type="csv", total_rows=2000)
    assert cfg["dataset"]["rows_charted"] == 2000
    assert cfg["dataset"]["charts_scope"] == "full_file"
    texts = [c.get("insight") or "" for c in cfg["charts"]]
    assert any("full file" in t for t in texts)
    assert not any("in the sample" in t for t in texts)


def test_missing_full_file_keeps_the_sample_charts(monkeypatch):
    monkeypatch.setattr(va, "viz_llm", None)
    rows = _rows(300)
    cfg = va.build_visualization_config_from_sample("ds2", rows, full_data_path="/nonexistent.csv")
    assert cfg["dataset"]["charts_scope"] == "sample"


def test_to_float_accepts_numpy_numbers():
    assert va._to_float(np.int64(7)) == 7.0
    assert va._to_float(np.float32(1.5)) == 1.5
    assert va._to_float(np.float64("nan")) is None