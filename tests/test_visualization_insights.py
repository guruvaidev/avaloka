"""Tests for the Auto Insights changes in app/agents/visualization_agent.py.

Covers the two fixes from the OpenRouter / Auto Insights work:
  1. Request routing: on OpenRouter, reasoning effort and host sorting go in
     extra_body, and a timeout is not retried in plain mode.
  2. Domain-aware insights: the profiling result feeds the prompt context,
     column stats, the profile overview and the upload chat message, and
     everything still works when profiling failed or was skipped.

No model is called: module settings are patched per test and the LLM is
replaced with a fake. Run with:

    pytest tests/test_visualization_insights.py -q
"""

from __future__ import annotations

import random

import pytest

from app.agents import visualization_agent as va


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROFILE = {
    "status": "success",
    "domain": {"category": "Healthcare", "reasoning": "Patient admissions with billing and conditions."},
    "column_explanations": {
        "Age": "Patient age in years at admission.",
        "Billing Amount": "Amount billed for the stay, in USD.",
        "Medical Condition": "Primary diagnosis for the admission.",
        "Blood Type": "Patient blood group.",
    },
    "predictive_values": {"Billing Amount": "high", "Medical Condition": "high", "Blood Type": "low"},
    "quick_insights": {
        "data_readiness_score": 78.4,
        "readiness_reasoning": "Few missing values; some negative billing amounts.",
        "red_flags": ["108 negative billing amounts", "Duplicate patient names"],
        "quick_wins": ["Drop negative bills"],
    },
    "data_quality_insights": {
        "strengths": ["No missing values"],
        "concerns": ["Duplicate patient names", "Room Number looks like an ID"],
    },
    "analysis_suggestions": ["Compare billing by medical condition", "Billing by admission type"],
}


def _count_chart(column, counts, chart_type="bar", source="data", even=True):
    insight = (f"The groups are nearly equal in size, so {column} on its own does not separate the data much."
               if even else f"{column} differs a lot between groups.")
    enc = {"color": {"field": column}} if chart_type == "pie" else {"x": {"field": column}}
    return {
        "title": f"Records by {column}",
        "type": chart_type,
        "insight": insight,
        "insight_source": source,
        "encodings": enc,
        "derived_data": {"x_key": column, "y_key": "count",
                         "points": [{column: f"g{i}", "count": c} for i, c in enumerate(counts)]},
    }


def _rows(n=120, seed=7):
    rnd = random.Random(seed)
    conditions = ["Cancer", "Obesity", "Diabetes", "Asthma", "Arthritis", "Hypertension"]
    blood = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
    admission = ["Elective", "Urgent", "Emergency"]
    return [{
        "Name": f"Patient {i}",
        "Age": rnd.randint(18, 85),
        "Medical Condition": rnd.choice(conditions),
        "Blood Type": rnd.choice(blood),
        "Admission Type": rnd.choice(admission),
        "Billing Amount": round(rnd.uniform(-500, 50000), 2),
        "Date of Admission": f"2023-{rnd.randint(1, 12):02d}-{rnd.randint(1, 28):02d}",
    } for i in range(n)]


@pytest.fixture
def openrouter(monkeypatch):
    """Module state as it is when requests really go to OpenRouter with gpt-oss."""
    monkeypatch.setattr(va, "_VIZ_PROVIDER", "openrouter")
    monkeypatch.setattr(va, "_VIZ_IS_REASONING", True)
    monkeypatch.setattr(va, "_VIZ_OPENROUTER_SORT", "throughput")
    monkeypatch.setattr(va, "_VIZ_SORT_ACTIVE", True)


@pytest.fixture
def groq(monkeypatch):
    monkeypatch.setattr(va, "_VIZ_PROVIDER", "groq")
    monkeypatch.setattr(va, "_VIZ_IS_REASONING", True)
    monkeypatch.setattr(va, "_VIZ_SORT_ACTIVE", False)


class FakeLLM:
    """Records each invoke; raises the queued exceptions first."""

    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = []

    def invoke(self, prompt, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


# ---------------------------------------------------------------------------
# 1. Request routing
# ---------------------------------------------------------------------------

def test_openrouter_sends_reasoning_and_provider_sort_in_extra_body(openrouter):
    kw = va._viz_request_kwargs("high")
    assert kw["max_tokens"] == va._VIZ_MAX_TOKENS
    assert kw["extra_body"] == {"reasoning": {"effort": "high"}, "provider": {"sort": "throughput"}}
    assert "reasoning_effort" not in kw          # Groq's flag must never reach OpenRouter


def test_openrouter_sort_can_be_turned_off(openrouter, monkeypatch):
    monkeypatch.setattr(va, "_VIZ_SORT_ACTIVE", False)
    kw = va._viz_request_kwargs("medium")
    assert kw["extra_body"] == {"reasoning": {"effort": "medium"}}


def test_openrouter_non_reasoning_model_sends_only_the_sort(openrouter, monkeypatch):
    monkeypatch.setattr(va, "_VIZ_IS_REASONING", False)
    assert va._viz_request_kwargs("high")["extra_body"] == {"provider": {"sort": "throughput"}}


def test_openrouter_with_nothing_to_send_has_no_extra_body(openrouter, monkeypatch):
    monkeypatch.setattr(va, "_VIZ_IS_REASONING", False)
    monkeypatch.setattr(va, "_VIZ_SORT_ACTIVE", False)
    assert va._viz_request_kwargs("high") == {"max_tokens": va._VIZ_MAX_TOKENS}


def test_groq_uses_top_level_reasoning_effort_and_no_extra_body(groq):
    kw = va._viz_request_kwargs("low")
    assert kw["reasoning_effort"] == "low"
    assert "extra_body" not in kw


def test_resolve_provider_follows_build_chat_model_for_default_backend(monkeypatch):
    monkeypatch.setattr(va, "_VIZ_BACKEND", "groq")
    monkeypatch.setattr(va, "resolve_provider", lambda **kw: "openrouter")
    assert va._resolve_viz_provider() == "openrouter"


def test_resolve_provider_uses_explicit_backend(monkeypatch):
    monkeypatch.setattr(va, "_VIZ_BACKEND", "local")
    monkeypatch.setattr(va, "resolve_provider", lambda **kw: pytest.fail("must not be called"))
    assert va._resolve_viz_provider() == "local"


def test_resolve_provider_never_raises(monkeypatch):
    def boom(**kw):
        raise ValueError("no keys")
    monkeypatch.setattr(va, "_VIZ_BACKEND", "groq")
    monkeypatch.setattr(va, "resolve_provider", boom)
    assert va._resolve_viz_provider() == "groq"


class APITimeoutError(Exception):
    """Same class name the openai client raises."""


@pytest.mark.parametrize("exc", [TimeoutError("read timed out"), APITimeoutError("timeout")])
def test_is_timeout_detects_timeouts(exc):
    assert va._is_timeout(exc) is True


def test_is_timeout_ignores_other_errors():
    assert va._is_timeout(ValueError("unexpected keyword 'extra_body'")) is False


def test_timeout_is_not_retried_in_plain_mode(openrouter, monkeypatch):
    llm = FakeLLM(TimeoutError("timed out"))
    monkeypatch.setattr(va, "viz_llm", llm)
    with pytest.raises(TimeoutError):
        va.invoke_viz_llm("prompt", effort="high")
    assert len(llm.calls) == 1                   # before the fix: 2 calls, double the wait


def test_auth_error_is_not_retried(openrouter, monkeypatch):
    llm = FakeLLM(RuntimeError("User not found"))
    monkeypatch.setattr(va, "viz_llm", llm)
    with pytest.raises(RuntimeError):
        va.invoke_viz_llm("prompt")
    assert len(llm.calls) == 1


def test_rejected_kwargs_are_retried_once_without_them(openrouter, monkeypatch):
    llm = FakeLLM(ValueError("unknown parameter: provider"))
    monkeypatch.setattr(va, "viz_llm", llm)
    assert va.invoke_viz_llm("prompt", effort="high") == "ok"
    assert len(llm.calls) == 2
    assert "extra_body" in llm.calls[0]
    assert llm.calls[1] == {}


def test_invoke_passes_the_routing_kwargs(openrouter, monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(va, "viz_llm", llm)
    va.invoke_viz_llm("prompt", effort="medium")
    assert llm.calls[0]["extra_body"]["provider"] == {"sort": "throughput"}
    assert llm.calls[0]["extra_body"]["reasoning"] == {"effort": "medium"}


def test_invoke_without_llm_raises_clearly(monkeypatch):
    monkeypatch.setattr(va, "viz_llm", None)
    with pytest.raises(RuntimeError, match="not configured"):
        va.invoke_viz_llm("prompt")


# ---------------------------------------------------------------------------
# 2. Profiling context for the insight prompt
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, {}, "text", {"status": "error"}, {"status": "skipped"}])
def test_dataset_context_is_none_without_a_usable_profile(bad):
    assert va._dataset_context(bad) is None


def test_dataset_context_carries_domain_meanings_and_issues():
    ctx = va._dataset_context(PROFILE)
    assert ctx["domain"] == "Healthcare"
    assert "Patient admissions" in ctx["domain_reasoning"]
    assert ctx["column_meanings"]["Age"].startswith("Patient age")
    # red flags first, concerns appended without repeating one already listed
    assert ctx["known_issues"] == ["108 negative billing amounts", "Duplicate patient names",
                                   "Room Number looks like an ID"]


def test_dataset_context_skips_unknown_domain():
    ctx = va._dataset_context({"domain": {"category": "Unknown"}, "column_explanations": {"a": "x"}})
    assert "domain" not in ctx
    assert ctx["column_meanings"] == {"a": "x"}


def test_dataset_context_clips_long_meanings():
    ctx = va._dataset_context({"column_explanations": {"a": "word " * 200}})
    assert len(ctx["column_meanings"]["a"]) <= va._CONTEXT_MEANING_CHARS


def test_dataset_context_caps_known_issues():
    many = {"quick_insights": {"red_flags": [f"flag {i}" for i in range(20)]}}
    assert len(va._dataset_context(many)["known_issues"]) == va._CONTEXT_MAX_ISSUES


def test_context_for_columns_keeps_only_requested_meanings():
    ctx = va._dataset_context(PROFILE)
    part = va._context_for_columns(ctx, ["Age", "Blood Type", None])
    assert set(part["column_meanings"]) == {"Age", "Blood Type"}
    assert part["domain"] == "Healthcare"
    assert part["known_issues"] == ctx["known_issues"]
    assert len(ctx["column_meanings"]) == 4      # the original is not modified


def test_context_for_columns_without_context_is_none():
    assert va._context_for_columns(None, ["Age"]) is None
    assert va._context_for_columns({}, ["Age"]) is None


# ---------------------------------------------------------------------------
# 3. Merging "nearly equal groups" charts
# ---------------------------------------------------------------------------

def test_merge_keeps_first_even_chart_and_reports_the_others():
    charts = [
        _count_chart("Medical Condition", [9300, 9200], even=False),
        _count_chart("Admission Type", [18655, 18576, 18269]),
        _count_chart("Blood Type", [6969, 6956, 6947], chart_type="pie"),
        _count_chart("Gender", [27800, 27700]),
        _count_chart("Insurance", [11200, 11100], even=False),
    ]
    kept, dropped = va._merge_even_count_charts(charts)
    titles = [c["title"] for c in kept]
    assert "Records by Admission Type" in titles            # first even chart stays
    assert "Records by Blood Type" not in titles
    assert "Records by Gender" not in titles
    assert dropped == ["Blood Type", "Gender"]               # pie column read from color
    assert [c["rank"] for c in kept] == list(range(1, len(kept) + 1))


def test_merge_never_goes_below_three_charts():
    charts = [_count_chart(c, [10, 10]) for c in ("A", "B", "C")]
    kept, dropped = va._merge_even_count_charts(charts)
    assert len(kept) == 3 and dropped == []


def test_merge_drops_only_as_many_as_room_allows():
    charts = [_count_chart(c, [10, 10]) for c in ("A", "B", "C", "D")]
    kept, dropped = va._merge_even_count_charts(charts)
    assert len(kept) == 3 and dropped == ["B"]


def test_merge_ignores_llm_written_and_non_count_charts():
    llm_chart = _count_chart("A", [10, 10], source="llm")
    hist = _count_chart("B", [10, 10])
    hist["type"] = "histogram"
    charts = [llm_chart, hist, _count_chart("C", [10, 10]), _count_chart("D", [5, 9], even=False)]
    kept, dropped = va._merge_even_count_charts(charts)
    assert dropped == [] and len(kept) == 4


def test_merge_with_a_single_even_chart_changes_nothing():
    charts = [_count_chart("A", [10, 10]), _count_chart("B", [1, 9], even=False)]
    assert va._merge_even_count_charts(charts) == (charts, [])


# ---------------------------------------------------------------------------
# 4. Column stats, profile overview, upload message
# ---------------------------------------------------------------------------

def test_enrich_column_stats_adds_meaning_and_predictive_value():
    stats = [{"name": "Age"}, {"name": "Blood Type"}, {"name": "Room Number"}]
    va._enrich_column_stats(stats, PROFILE)
    assert stats[0]["meaning"] == "Patient age in years at admission."
    assert "predictive_value" not in stats[0]
    assert stats[1]["predictive_value"] == "low"
    assert stats[2] == {"name": "Room Number"}


def test_enrich_column_stats_without_profile_is_a_no_op():
    stats = [{"name": "Age"}]
    va._enrich_column_stats(stats, None)
    assert stats == [{"name": "Age"}]


@pytest.mark.parametrize("bad", [None, {"status": "error"}, {"status": "skipped"}, []])
def test_profile_overview_is_none_without_a_usable_profile(bad):
    assert va.build_profile_overview(bad) is None


def test_profile_overview_shape():
    ov = va.build_profile_overview(PROFILE)
    assert ov["domain"] == "Healthcare"
    assert ov["readiness_score"] == 78                     # rounded
    assert ov["red_flags"][0] == "108 negative billing amounts"
    assert ov["high_value_columns"] == ["Billing Amount", "Medical Condition"]
    assert ov["analysis_suggestions"][0] == "Compare billing by medical condition"
    for key in ("quick_wins", "strengths", "concerns", "incomplete_sections"):
        assert isinstance(ov[key], list)


def test_profile_overview_tolerates_bad_values():
    ov = va.build_profile_overview({"domain": {"category": "Unknown"},
                                    "quick_insights": {"data_readiness_score": "n/a", "red_flags": "not a list"}})
    assert ov["domain"] is None
    assert ov["readiness_score"] is None
    assert ov["red_flags"] == []


def test_upload_message_uses_total_rows_not_the_sample():
    charts = [{"headline": "Obesity patients are billed the most", "insight": "long text"}]
    msg = va.build_upload_message(va.build_profile_overview(PROFILE), charts,
                                  total_rows=55500, n_columns=15)
    assert "55,500 rows × 15 columns" in msg
    assert "**Healthcare** dataset" in msg
    assert "**Data readiness: 78/100.**" in msg
    assert "- Obesity patients are billed the most" in msg      # headline preferred over insight
    assert "**Watch out for**" in msg and "108 negative billing amounts" in msg
    assert "**Try asking**" in msg


def test_upload_message_without_profile_still_reads_well():
    msg = va.build_upload_message(None, [], total_rows=None, n_columns=7)
    assert msg == "Dataset · 7 columns."


def test_upload_message_lists_at_most_three_findings():
    charts = [{"insight": f"finding {i}"} for i in range(6)]
    msg = va.build_upload_message(None, charts, total_rows=100, n_columns=3)
    assert msg.count("- finding") == 3


def test_upload_message_falls_back_to_concerns_and_quick_wins():
    ov = {"concerns": ["Duplicate names"], "quick_wins": ["Drop negative bills"]}
    msg = va.build_upload_message(ov, [], total_rows=10, n_columns=2)
    assert "- Duplicate names" in msg and "- Drop negative bills" in msg


# ---------------------------------------------------------------------------
# 5. The whole builder, heuristic path (no LLM)
# ---------------------------------------------------------------------------

@pytest.fixture
def no_llm(monkeypatch):
    monkeypatch.setattr(va, "viz_llm", None)


def test_builder_with_profile_adds_overview_message_and_context(no_llm):
    cfg = va.build_visualization_config_from_sample(
        dataset_id="ds-test", sample_rows=_rows(), total_rows=55500, profiling_result=PROFILE)
    assert cfg["visualization_status"] == "ready"
    assert cfg["dataset_context"]["domain"] == "Healthcare"
    assert cfg["profile_overview"]["domain"] == "Healthcare"
    assert cfg["upload_message"].startswith("**Healthcare** dataset · 55,500 rows")
    assert cfg["dataset"]["total_rows"] == 55500
    assert cfg["dataset"]["rows_sampled"] == 120
    age = next(c for c in cfg["column_stats"] if c["name"] == "Age")
    assert age["meaning"] == "Patient age in years at admission."
    assert cfg["insight_model"]["used"] is False


def test_builder_without_profile_still_returns_charts(no_llm):
    cfg = va.build_visualization_config_from_sample(dataset_id="ds-test", sample_rows=_rows())
    assert cfg["visualization_status"] == "ready"
    assert cfg["charts"], "heuristic charts expected when there is no LLM and no profile"
    assert cfg["profile_overview"] is None
    assert "dataset_context" not in cfg
    assert cfg["upload_message"].startswith("Dataset · ")


def test_builder_with_failed_profile_behaves_like_no_profile(no_llm):
    cfg = va.build_visualization_config_from_sample(
        dataset_id="ds-test", sample_rows=_rows(), profiling_result={"status": "error"})
    assert cfg["profile_overview"] is None
    assert "dataset_context" not in cfg


def test_builder_reports_the_routing_it_used(no_llm, openrouter):
    cfg = va.build_visualization_config_from_sample(dataset_id="ds-test", sample_rows=_rows(40))
    model = cfg["insight_model"]
    assert model["provider"] == "openrouter"
    assert model["openrouter_sort"] == "throughput"


# ---------------------------------------------------------------------------
# 6. Column statistics computed on the whole file
# ---------------------------------------------------------------------------

def _write_csv(tmp_path, rows):
    import csv
    path = tmp_path / "data.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def test_column_stats_use_the_full_file_not_the_sample(no_llm, tmp_path):
    rows = _rows(2000)
    rows[1500]["Billing Amount"] = -9999.0         # only in the part outside the sample
    rows[1600]["Blood Type"] = "Rare"
    path = _write_csv(tmp_path, rows)
    sample = rows[:200]
    cfg = va.build_visualization_config_from_sample(
        dataset_id="ds-test", sample_rows=sample, full_data_path=path, full_data_type="csv",
        total_rows=2000)
    assert cfg["column_stats_scope"] == "full_file"
    stats = {c["name"]: c for c in cfg["column_stats"]}
    billing = stats["Billing Amount"]
    assert billing["min"] == -9999.0
    assert billing["n_unique"] == len({r["Billing Amount"] for r in rows})
    assert stats["Blood Type"]["n_unique"] == 9                # 8 groups + "Rare"
    top = stats["Blood Type"]["top_values"]
    assert all(t["count"] > 30 for t in top)                    # full-file counts, not sample counts


def test_full_stats_keep_the_sample_types_and_roles(no_llm, tmp_path):
    rows = _rows(500)
    path = _write_csv(tmp_path, rows)
    cfg = va.build_visualization_config_from_sample(
        dataset_id="ds-test", sample_rows=rows[:100], full_data_path=path, full_data_type="csv")
    by_name = {c["name"]: c for c in cfg["columns"]}
    for row in cfg["column_stats"]:
        assert row["dtype"] == by_name[row["name"]]["dtype"]
        assert row["role"] == by_name[row["name"]]["role"]


def test_full_stats_missing_values_match_is_missing(tmp_path):
    rows = [{"a": v, "b": i} for i, v in enumerate(["x", "", "NA", "null", "None", "nan", "y", "x"])]
    path = _write_csv(tmp_path, rows)
    profiles = va.profile_columns([{k: str(v) for k, v in r.items()} for r in rows])
    stats = {c["name"]: c for c in va.build_column_stats_full(path, "csv", profiles)}
    assert stats["a"]["missing_pct"] == 62.5                   # 5 of 8 are missing tokens
    assert stats["a"]["n_unique"] == 2


def test_full_stats_dates(tmp_path):
    rows = [{"d": f"2024-0{m}-1{m}", "n": m} for m in range(1, 10)] * 3
    path = _write_csv(tmp_path, rows)
    profiles = va.profile_columns(rows)
    d = next(c for c in va.build_column_stats_full(path, "csv", profiles) if c["name"] == "d")
    assert (d["min"], d["max"], d["n_months"], d["n_unique"]) == ("2024-01-11", "2024-09-19", 9, 9)


def test_full_stats_fall_back_to_sample_without_a_file(no_llm):
    cfg = va.build_visualization_config_from_sample(dataset_id="ds-test", sample_rows=_rows(50))
    assert cfg["column_stats_scope"] == "sample"
    assert cfg["column_stats"]


def test_full_stats_fall_back_when_file_too_big(no_llm, tmp_path, monkeypatch):
    path = _write_csv(tmp_path, _rows(300))
    monkeypatch.setattr(va, "_FULL_STATS_MAX_BYTES", 10)
    cfg = va.build_visualization_config_from_sample(
        dataset_id="ds-test", sample_rows=_rows(300)[:50], full_data_path=path, full_data_type="csv")
    assert cfg["column_stats_scope"] == "sample"


def test_full_stats_none_when_a_column_is_missing(tmp_path):
    path = _write_csv(tmp_path, [{"a": 1}, {"a": 2}, {"a": 3}])
    profiles = va.profile_columns([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}])
    assert va.build_column_stats_full(path, "csv", profiles) is None


# ---------------------------------------------------------------------------
# 7. Profile warnings checked against the data
# ---------------------------------------------------------------------------

ICU_STATS = [
    {"name": "ethnicity", "dtype": "categorical", "missing_pct": 1.52,
     "top_values": [{"value": "Caucasian", "count": 70684, "pct": 78.3},
                    {"value": "African American", "count": 9547, "pct": 10.6}]},
    {"name": "bmi", "dtype": "numeric", "missing_pct": 3.74, "min": 14.8, "max": 67.8},
    {"name": "weight", "dtype": "numeric", "missing_pct": 2.66, "min": 38.6, "max": 186.0},
    {"name": "map_apache", "dtype": "numeric", "missing_pct": 1.08, "min": 40.0, "max": 200.0},
    {"name": "age", "dtype": "numeric", "missing_pct": 4.61, "min": 16.0, "max": 89.0},
]
ICU_BY_NAME = {s["name"]: s for s in ICU_STATS}


@pytest.mark.parametrize("claim, expected", [
    ("Ethnicity >90% Caucasian", "Ethnicity 78.3% Caucasian"),
    ("BMI/weight missing for ~2% of records", "BMI/weight missing for 2.7–3.7% of records"),
    ("MAP values as low as 44 mmHg", "MAP values as low as 40 mmHg"),
])
def test_wrong_profile_numbers_are_corrected(claim, expected):
    assert va._verify_claim(claim, ICU_BY_NAME)[0] == expected


@pytest.mark.parametrize("claim", [
    "Age missing for about 4.6% of patients",                     # right
    "Ethnicity is heavily skewed toward Caucasian (78% of rows)",  # rounded, still right
    "Over 75% of patients are Caucasian",                         # names no column: not checkable
    "age up to 89 years",                                         # right extreme
    "Impute missing age values",                                  # no number
])
def test_right_or_uncheckable_claims_are_kept(claim):
    fixed, notes = va._verify_claim(claim, ICU_BY_NAME)
    assert fixed == claim and notes == []


def test_verify_profile_claims_corrects_overview_and_records_it():
    ov = {"red_flags": ["Ethnicity >90% Caucasian", "No duplicates"],
          "concerns": ["BMI/weight missing for ~2% of records"],
          "readiness_reasoning": "Few gaps."}
    out = va.verify_profile_claims(ov, ICU_STATS)
    assert out["red_flags"] == ["Ethnicity 78.3% Caucasian", "No duplicates"]
    assert out["concerns"] == ["BMI/weight missing for 2.7–3.7% of records"]
    assert [c["field"] for c in out["corrections"]] == ["red_flags", "concerns"]


def test_upload_message_shows_the_corrected_number():
    ov = va.verify_profile_claims({"red_flags": ["Ethnicity >90% Caucasian"]}, ICU_STATS)
    msg = va.build_upload_message(ov, [], total_rows=91713, n_columns=84)
    assert "78.3% Caucasian" in msg and "90%" not in msg


def test_known_issues_sent_to_the_model_are_corrected():
    ctx = va.verify_known_issues({"known_issues": ["Ethnicity >90% Caucasian"]}, ICU_STATS)
    assert ctx["known_issues"] == ["Ethnicity 78.3% Caucasian"]


def test_verify_without_stats_or_overview_is_a_no_op():
    assert va.verify_profile_claims(None, ICU_STATS) is None
    ov = {"red_flags": ["x"]}
    assert va.verify_profile_claims(ov, []) == {"red_flags": ["x"]}


# ---------------------------------------------------------------------------
# 8. Outcome rates instead of shares, Yes/No labels
# ---------------------------------------------------------------------------

def _icu_rows(n=1500, seed=3):
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        eth = rnd.choice(["Caucasian"] * 7 + ["African American", "Hispanic", "Asian"])
        arf = 1 if rnd.random() < 0.1 else 0
        death = 1 if rnd.random() < (0.25 if arf else 0.07) else 0
        rows.append({"encounter_id": 100000 + i, "age": rnd.randint(18, 90), "ethnicity": eth,
                     "icu_type": rnd.choice(["MICU", "CSICU", "Med-Surg ICU", "Neuro ICU"]),
                     "arf_apache": arf, "d1_spo2_min": rnd.randint(60, 100), "hospital_death": death})
    return rows


def test_rate_by_yes_no_group_uses_yes_and_no():
    rows = _icu_rows()
    points, meta, text = va._ground_rate_by_group(rows, "arf_apache", "hospital_death")
    assert {p["x"] for p in points} == {"Yes", "No"}
    assert "highest for Yes" in text and "1.0" not in text


def test_share_chart_becomes_outcome_rate_when_outcome_exists():
    rows = _icu_rows()
    profiles = va.profile_columns(rows)
    by_name = {p["name"]: p for p in profiles}
    share = va._ground_all([{"title": "Share of records by ethnicity", "type": "pie",
                             "encodings": {"theta": {"field": "count", "aggregate": "sum"},
                                           "color": {"field": "ethnicity"}},
                             "config": {"top_k": 5}, "derived_data": {}}], rows, by_name)
    out = va._shares_to_outcome_rates(share, profiles, by_name, rows)
    assert len(out) == 1
    assert out[0]["title"] == "hospital_death rate by ethnicity"
    assert out[0]["replaced_share_chart"] == "Share of records by ethnicity"
    assert out[0]["derived_data"]["points"]


def test_share_chart_dropped_when_rate_by_same_column_exists():
    rows = _icu_rows()
    profiles = va.profile_columns(rows)
    by_name = {p["name"]: p for p in profiles}
    rate = va._ground_all([va._rate_chart("hospital_death", "icu_type", "categorical")], rows, by_name)
    share = va._ground_all([_count_chart("icu_type", [1, 1])], rows, by_name)
    out = va._shares_to_outcome_rates(rate + share, profiles, by_name, rows)
    assert [c["title"] for c in out] == ["hospital_death rate by icu_type"]


def test_shares_kept_when_no_outcome_column():
    rows = _rows(300)                    # healthcare-style data, no 0/1 outcome
    profiles = va.profile_columns(rows)
    by_name = {p["name"]: p for p in profiles}
    charts = va._ground_all([_count_chart("Blood Type", [1, 1])], rows, by_name)
    assert va._shares_to_outcome_rates(charts, profiles, by_name, rows) == charts


def test_builder_on_outcome_data_keeps_meaning_and_predictive_value(no_llm):
    profile = {"status": "success", "domain": {"category": "Healthcare/Critical Care"},
               "column_explanations": {"age": "Patient's age in years at admission.",
                                       "ethnicity": "Patient ethnicity."},
               "predictive_values": {"age": "high", "ethnicity": "low"},
               "quick_insights": {"red_flags": ["Ethnicity >90% Caucasian"]}}
    cfg = va.build_visualization_config_from_sample(
        dataset_id="icu", sample_rows=_icu_rows(), profiling_result=profile)
    stats = {c["name"]: c for c in cfg["column_stats"]}
    assert stats["age"]["meaning"] == "Patient's age in years at admission."
    assert stats["age"]["predictive_value"] == "high"
    assert stats["ethnicity"]["predictive_value"] == "low"
    flag = cfg["profile_overview"]["red_flags"][0]
    assert "90%" not in flag and "Caucasian" in flag
    for ch in cfg["charts"]:                       # no plain share chart of a category
        field = va._share_chart_field(ch)
        assert not (field and by_dtype(cfg, field) == "categorical"), ch["title"]


def by_dtype(cfg, name):
    return next((c["dtype"] for c in cfg["columns"] if c["name"] == name), None)


# ---------------------------------------------------------------------------
# 9. Fast correlation gives the same answers
# ---------------------------------------------------------------------------

def test_fast_correlation_matches_the_original():
    rows = _rows(500)
    rows[3]["Age"] = None
    rows[7]["Billing Amount"] = "n/a"
    for a, b in [("Age", "Billing Amount"), ("Billing Amount", "Age"), ("Age", "Age")]:
        slow = va._pearson(va._paired(rows, a, b))
        fast = va._corr(rows, a, b)
        assert (slow is None and fast is None) or abs(slow - fast) < 1e-9


def test_fast_correlation_handles_constant_and_tiny_columns():
    rows = [{"a": 1, "b": 5}, {"a": 2, "b": 5}, {"a": 3, "b": 5}]
    assert va._corr(rows, "a", "b") is None           # constant column
    assert va._corr(rows[:2], "a", "b") is None        # fewer than 3 pairs


def test_fast_correlation_never_mixes_row_sets():
    a = [{"x": i, "y": i} for i in range(10)]
    b = [{"x": i, "y": -i} for i in range(10)]
    assert va._corr(a, "x", "y") == pytest.approx(1.0)
    assert va._corr(b, "x", "y") == pytest.approx(-1.0)
    assert va._corr(a, "x", "y") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 10. Model headlines checked against the chart (cases from the ICU upload)
# ---------------------------------------------------------------------------

ADMIT = [{"x": "Other ICU", "y": 14.4}, {"x": "Other Hospital", "y": 12.0},
         {"x": "Accident & Emergency", "y": 11.2}, {"x": "Floor", "y": 10.1},
         {"x": "Operating Room / Recovery", "y": 3.7}]
POSTOP = [{"x": "No", "y": 9.8}, {"x": "Yes", "y": 3.9}]
INTUB = [{"x": "Yes", "y": 20.1}, {"x": "No", "y": 6.6}]
ICU = [{"x": "MICU", "y": 12.1}, {"x": "Neuro ICU", "y": 8.0}, {"x": "CSICU", "y": 5.5}]


@pytest.mark.parametrize("text, points, field", [
    # Seen on 10 Oct: OR/Recovery is the LOWEST group.
    ("Patients admitted from the operating room/recovery have a higher death rate than those "
     "coming from the emergency department.", ADMIT, "icu_admit_source"),
    # Seen on 10 Oct: post-operative (Yes) is 3.9% vs 9.8%.
    ("Post-operative admissions carry a higher risk of hospital death compared with "
     "non-post-operative cases.", POSTOP, "apache_post_operative"),
    ("CSICU has a higher death rate than MICU.", ICU, "icu_type"),
])
def test_wrong_headlines_are_caught(text, points, field):
    assert va._llm_claim_conflict(text, "bar", points, "", x_field=field)


@pytest.mark.parametrize("text, points, field", [
    ("Being intubated on ICU admission is associated with a substantially higher hospital "
     "death rate compared with not being intubated.", INTUB, "intubated_apache"),
    ("Non-post-operative patients have a higher death rate than post-operative patients.",
     POSTOP, "apache_post_operative"),
    ("MICU patients die more often than CSICU patients.", ICU, "icu_type"),
    ("Hospital death rate differs across ICU types, with some units showing noticeably "
     "higher mortality.", ICU, "icu_type"),
    ("Patients from Other ICU have the highest death rate.", ADMIT, "icu_admit_source"),
    ("Operating Room / Recovery admissions have the lowest death rate.", ADMIT, "icu_admit_source"),
])
def test_right_headlines_are_kept(text, points, field):
    assert va._llm_claim_conflict(text, "bar", points, "", x_field=field) is None


def test_group_names_match_despite_spacing_and_punctuation():
    assert list(va._label_spans("Operating Room / Recovery", "from the operating room/recovery unit"))
    assert list(va._label_spans("Accident & Emergency", "accident and emergency")) == []


def test_false_no_outcome_claim_is_dropped_when_outcome_exists():
    stats = ICU_STATS + [{"name": "hospital_death", "dtype": "numeric", "missing_pct": 0.0,
                          "min": 0, "max": 1}]
    ov = {"red_flags": ["Null BMI/weight entries", "Ethnicity homogeneity", "No explicit outcome column"]}
    out = va.verify_profile_claims(ov, stats)
    assert out["red_flags"] == ["Null BMI/weight entries", "Ethnicity homogeneity"]
    assert out["corrections"][0]["reason"] == "hospital_death is a yes/no outcome column"


def test_no_outcome_claim_kept_when_there_is_none():
    ov = {"red_flags": ["No explicit outcome column"]}
    assert va.verify_profile_claims(ov, ICU_STATS)["red_flags"] == ["No explicit outcome column"]


@pytest.mark.parametrize("text, matches", [
    ("No explicit outcome column", True),
    ("Dataset lacks a clear target variable", True),
    ("No missing values in the outcome column", False),
    ("Missing outcome values for 2% of rows", False),
    ("No outcomes recorded for transfers", False),
    # Seen on 10 Oct (three words between "No" and "outcome"):
    ("No mortality or discharge outcome column.", True),
    ("No explicit outcome variable (e.g., mortality).", True),
    ("No missing or null values in the outcome column", False),
])
def test_no_outcome_wording(text, matches):
    assert bool(va._NO_OUTCOME_RE.search(text)) is matches


ICU_TYPES = [{"x": "MICU", "y": 12.1}, {"x": "Med-Surg ICU", "y": 8.9},
             {"x": "Neuro ICU", "y": 7.9}, {"x": "CSICU", "y": 5.5}]


def test_plural_group_names_are_matched():
    # Seen on 10 Oct: MICU is the highest group, not Med-Surg ICU.
    text = "Med-Surg ICUs show a higher hospital death rate compared to MICUs and other ICU types."
    assert va._llm_claim_conflict(text, "bar", ICU_TYPES, "", x_field="icu_type")


def test_plural_true_claim_is_kept():
    text = "MICUs show a higher hospital death rate than CSICUs."
    assert va._llm_claim_conflict(text, "bar", ICU_TYPES, "", x_field="icu_type") is None


def test_summary_calls_flags_yes_no_columns_and_lists_the_outcome_first():
    rows = _icu_rows(400)
    profiles = va.profile_columns(rows)
    summary = va.build_dataset_summary(profiles, rows)
    first = summary["insights"][0]
    assert "yes/no column" in first and "yes/no outcome" not in first
    assert first.index("hospital_death") < first.index("arf_apache")


# ---------------------------------------------------------------- CI follow-ups

class _Resp:
    def __init__(self, status_code):
        self.status_code = status_code


class _ProviderError(Exception):
    def __init__(self, message, response=None):
        super().__init__(message)
        self.response = response


@pytest.mark.parametrize("error", [
    _ProviderError("Too Many Requests", response=_Resp(429)),
    RuntimeError("Rate limit reached for model openai/gpt-oss-120b"),
    type("RateLimitError", (Exception,), {})("slow down"),
])
def test_a_rate_limit_in_any_shape_is_not_retried(monkeypatch, error):
    calls = []

    class LLM:
        def invoke(self, prompt, **kwargs):
            calls.append(kwargs)
            raise error

    monkeypatch.setattr(va, "viz_llm", LLM())
    with pytest.raises(type(error)):
        va.invoke_viz_llm("prompt")
    assert len(calls) == 1


def test_a_rejected_kwarg_is_still_retried_plain(monkeypatch):
    calls = []

    class LLM:
        def invoke(self, prompt, **kwargs):
            calls.append(kwargs)
            if kwargs:
                raise RuntimeError("unknown parameter: extra_body")
            return "ok"

    monkeypatch.setattr(va, "viz_llm", LLM())
    assert va.invoke_viz_llm("prompt") == "ok"
    assert calls[1] == {}


@pytest.mark.parametrize("sentence", [
    "Older patients show higher death counts.",
    "The number of deaths rises with age.",
    "x",
    "see chart",
])
def test_a_rate_chart_drops_count_claims_and_stubs(sentence):
    assert va._rate_chart_insight(sentence)["insight"] is None


def test_a_rate_chart_keeps_a_real_rate_finding():
    out = va._rate_chart_insight("Death rate climbs steadily with age.")
    assert out["insight_source"] == "llm"
    assert out["insight"] == "Death rate climbs steadily with age."