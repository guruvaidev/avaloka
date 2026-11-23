"""B5 and the chart builders: the price/volume/mix bridge sums to ΔR exactly,
every builder emits a JSON-only option with no ``<``, ``=>`` or
``function`` in any string, and every chart has alt text and a data table.

B4: the retail and fintech templates on seeded fixtures, and a missing role
as one ``missing_template_data`` caveat naming the role.

Bites (run by hand, in the PR): set ``mix = 0.0`` and
``test_price_volume_mix_sum_to_delta`` goes red; skip the
``missing_template_data`` caveat and
``test_retail_missing_role_is_one_caveat_naming_the_role`` goes red.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from avaloka.outcomes import semantics as S
from avaloka.outcomes.recommend import recommend
from avaloka.outcomes.templates import basket_lift, build_chart, price_volume_mix, safe_label


@pytest.mark.parametrize("seed", range(25))
def test_price_volume_mix_sum_to_delta(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 40))
    p0, p1 = rng.uniform(1, 50, n), rng.uniform(1, 50, n)
    q0, q1 = rng.integers(0, 500, n), rng.integers(0, 500, n)
    b = price_volume_mix(p0, q0, p1, q1)
    assert b["price"] + b["volume_pure"] + b["mix"] == pytest.approx(b["delta"], rel=1e-12, abs=1e-9)
    assert b["price"] + b["volume"] == pytest.approx(b["delta"], rel=1e-12, abs=1e-9)
    assert b["mix"] != 0 or n == 1


def test_pure_price_change_is_all_price():
    b = price_volume_mix([10, 20], [5, 5], [11, 22], [5, 5])
    assert (b["price"], b["volume"], b["mix"]) == (15.0, 0.0, 0.0)


def test_basket_lift_floor_and_value():
    rows = []
    for o in range(100):
        rows += [(o, "bread"), (o, "butter")] if o < 40 else [(o, "bread")] if o < 70 else [(o, "jam")]
    out = basket_lift(pd.DataFrame(rows, columns=["order_id", "sku"]), "order_id", "sku")
    assert len(out) == 1 and out[0]["orders"] == 40
    assert out[0]["lift"] == pytest.approx((40 / 100) / ((70 / 100) * (40 / 100)))
    assert basket_lift(pd.DataFrame(rows[:58], columns=["order_id", "sku"]), "order_id", "sku") == []


def _walk_strings(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v)
    elif isinstance(obj, str):
        yield obj


PLANT = "<script>alert(1)</script> => function(){}"

rng = np.random.default_rng(5)
_N = 240
_DF = pd.DataFrame({
    "month": np.tile(pd.date_range("2025-01-01", periods=8, freq="MS"), _N // 8),
    "region": np.resize([PLANT, "north", "south"], _N),
    "revenue": rng.uniform(10, 100, _N).round(2),
    "units_sold_q": rng.integers(1, 50, _N),
    "state": np.resize(["CA", "NY", "TX", "WA"], _N),
    "from_step": np.resize(["ad", "site", PLANT], _N),
    "to_step": np.resize(["buy", "buy", "buy"], _N),
})
GOALS = ["revenue trend", "revenue breakdown by region", "distribution of revenue",
         "revenue vs units", "sales by state", "top regions", "journey from step to step"]


@pytest.mark.parametrize("goal", GOALS)
def test_every_candidate_builds_a_clean_option(goal):
    sem = S.infer(_DF)
    rec = recommend(goal, _DF, sem)
    assert rec.candidates, rec.refused
    for cand in rec.candidates:
        chart = build_chart(cand, rec.question_type, _DF, sem, title="t")
        json.dumps(chart["option"], allow_nan=False)
        for s in _walk_strings(chart["option"]):
            assert "<" not in s and "=>" not in s and "function" not in s, (cand.chart_kind, s)
        assert 20 <= len(chart["alt_text"]) <= 400
        assert chart["data_table"]["columns"] and len(chart["data_table"]["rows"]) <= 500


def test_safe_label_keeps_the_text_readable():
    assert safe_label("<b>") == "‹b›"
    assert "function" not in safe_label("my function")


def test_choropleth_embeds_no_suppressed_region():
    df = pd.DataFrame({"state": ["CA"] * 12 + ["NY"] * 11 + ["WY"] * 3, "sales": [1.0] * 26})
    sem = S.infer(df)
    rec = recommend("sales by state", df, sem)
    chart = build_chart(rec.candidates[0], "geo", df, sem, title="t")
    blob = json.dumps(chart)
    assert chart["suppressed_regions"] == 1 and chart["k_floor"] == 10
    assert '"WY"' not in blob and "Wyoming" not in blob
    assert chart["asset"] == "us_states_20m"


# ── B4: templates on fixtures ─────────────────────────────────────────────

import subprocess  # noqa: E402
import sys  # noqa: E402

from avaloka.outcomes import spec  # noqa: E402
from avaloka.outcomes.templates import build_report, choose_template, infer_audience  # noqa: E402

from .conftest import ANALYSIS, CREATED  # noqa: E402


def _build(df, goal="how are we doing", **kw):
    return build_report(df, goal, analysis=ANALYSIS, created_at=CREATED, **kw)


def test_template_choice_follows_the_domain():
    assert [choose_template(d) for d in ("finance", "retail", "marketing", "healthcare", None)] == \
        ["fintech_credit", "retail_performance", "generic", "generic", "generic"]


def test_retail_template_panels(retail_df):
    r = _build(retail_df)
    assert spec.is_valid(r)
    assert (r["domain"], r["template"]) == ("retail", "retail_performance")
    assert [k["id"] for k in r["kpis"]] == ["net_sales", "units", "avg_basket", "sell_through"]
    assert [c["id"] for c in r["charts"]] == ["yoy_bridge", "sell_through", "affinity"]
    assert [m["id"] for m in r["maps"]] == ["region_map"]
    assert not [c for c in r["caveats"] if c["kind"] == "missing_template_data"]
    assert r["completion"]["state"] == "answered"


def test_retail_missing_role_is_one_caveat_naming_the_role(retail_df):
    r = _build(retail_df.drop(columns=["units_received"]))
    missing = [c for c in r["caveats"] if c["kind"] == "missing_template_data"]
    assert len(missing) == 1
    assert "units received" in missing[0]["text"]
    assert "units_received" not in missing[0]["text"]          # a role, not a column
    assert "sell_through" not in [c["id"] for c in r["charts"]]
    assert r["completion"]["state"] == "partial"
    assert any(s["status"] == "needs_data" for s in r["steps"])


def test_fintech_template_panels(fintech_df):
    r = _build(fintech_df)
    assert spec.is_valid(r)
    assert (r["domain"], r["template"]) == ("finance", "fintech_credit")
    assert [k["id"] for k in r["kpis"]] == ["dq30", "dq60", "dq90", "nco_rate", "exposure", "hhi"]
    assert [c["id"] for c in r["charts"]] == ["vintage", "roll", "pnl", "concentration"]
    assert not any(c["kind"] == "currency_unstated" for c in r["caveats"])
    assert all(f.get("currency") == "USD" for f in r["figures"] if f["unit"] == "currency")


def test_fintech_without_dates_names_the_roles(fintech_df):
    r = _build(fintech_df.drop(columns=["origination_date", "as_of_date"]))
    [missing] = [c for c in r["caveats"] if c["kind"] == "missing_template_data"]
    assert "an origination date and an observation date" in missing["text"]
    assert "vintage" not in [c["id"] for c in r["charts"]]


def test_generic_template_is_the_recommender(retail_df):
    r = _build(retail_df[["order_date", "category", "quantity"]], template="generic")
    assert r["template"] == "generic" and r["charts"]
    assert spec.is_valid(r)


def test_every_figure_has_its_evidence(retail_df, fintech_df):
    for df in (retail_df, fintech_df):
        r = _build(df)
        assert spec.unresolved_refs(r) == []
        for f in r["figures"]:
            value, unit, currency = spec.evidence_leaf(r["evidence"], f["evidence_ref"])
            assert (value, unit, currency) == (f["value"], f["unit"], f.get("currency"))
        for ptr, text in spec.text_slots(r):
            assert spec.free_numbers(text) == [], (ptr, text)


def test_slots_replace_the_default_text(retail_df):
    r = _build(retail_df, slots={"headline": "Fixed headline {f1}.",
                                 "captions": {"region_map": "Fixed caption."}})
    assert r["headline"]["text"] == "Fixed headline {f1}."
    assert r["maps"][0]["caption"] == "Fixed caption."


def test_build_is_deterministic(retail_df):
    assert _build(retail_df) == _build(retail_df)


def test_audience_from_the_goal():
    assert infer_audience("which features drive churn in the model") == "analyst"
    assert infer_audience("how are sales doing") == "executive"


def test_no_model_is_imported():
    probe = ("import sys; sys.path.insert(0, 'tests/avaloka/outcomes'); import conftest as c; "
             "from avaloka.outcomes.templates import build_report; "
             "build_report(c.retail_frame(), 'x', analysis=c.ANALYSIS, created_at=c.CREATED); "
             "print(sorted(m for m in sys.modules if m.split('.')[0] in ('anthropic', 'openai', 'app')))")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parents[3]))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


#: Budgets are ratios to vectorised pandas work on the same frame, timed in the
#: same test, so they hold on a fast laptop and a shared CI runner alike. A
#: wall-clock 1.5 s failed on about 10% of CI runs (14 of 142 on the 2.0 line).
#: Measured in the gate image, 20 rounds, best of 5 (PR description has the data):
#:   build / baseline:        vectorised 0.92-1.42, per-order loop 1.61-1.96
#:   basket_lift / baseline:  vectorised 0.95-1.43, per-order loop 2.96-4.59
BUILD_BUDGET_RATIO = 2.0
BASKET_LIFT_BUDGET_RATIO = 2.2


def _best_of(n, fn, *args):
    import time

    best = float("inf")
    for _ in range(n):
        t = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t)
    return best


def _vectorised_baseline(df, key):
    """A fixed vectorised pandas workload over the frame, about as long as the build."""
    for _ in range(6):
        df.groupby(key, sort=False).size()
        for col in df.columns:
            df[col].value_counts(dropna=False)
        df.sort_values(key, kind="stable")
        keys = df[[key]].drop_duplicates().head(50_000)
        keys.merge(keys, on=key)


def _lift_baseline(df, order, product):
    """The same shape of work as basket_lift (dedupe, count, self-merge, group), no Python loop."""
    lines = df[[order, product]].dropna().drop_duplicates()
    lines["p"] = lines[product].astype(str)
    lines["p"].value_counts()
    pairs = lines[[order, "p"]].merge(lines[[order, "p"]], on=order)
    pairs.groupby(["p_x", "p_y"]).size()


@pytest.mark.slow
@pytest.mark.parametrize("which", ["retail", "fintech"])
def test_build_of_200k_rows_is_within_budget(which, request):
    """§6.3: the deterministic report build on a 200k-row sample stays within
    BUILD_BUDGET_RATIO of a fixed vectorised workload on the same frame, and
    (retail) basket_lift within BASKET_LIFT_BUDGET_RATIO of the same work done
    without a Python loop. Best of five each, interleaved.

    Bite (run in the PR): put back the per-order pair loop basket_lift had
    before 25b64089 and the retail basket_lift ratio goes to about 3-4.6.
    """
    from avaloka.outcomes import templates

    from .conftest import BUDGET_LINES, fintech_frame, retail_frame
    base = retail_frame() if which == "retail" else fintech_frame()
    reps = 200_000 // len(base) + 1
    df = pd.concat([base] * reps, ignore_index=True).head(200_000)
    key = "order_id" if which == "retail" else "account_id"
    df[key] = df[key] + "_" + (df.index // len(base)).astype(str)

    build_s = baseline_s = float("inf")
    for _ in range(5):
        baseline_s = min(baseline_s, _best_of(1, _vectorised_baseline, df, key))
        build_s = min(build_s, _best_of(1, _build, df))
    ratio = build_s / baseline_s
    lines = [f"OUTCOME-BUDGET {which} build ratio={ratio:.3f} k={BUILD_BUDGET_RATIO} "
             f"build_s={build_s:.3f} baseline_s={baseline_s:.3f}"]
    if which == "retail":  # measured before any assert, so a failure still reports both ratios
        lift_s = lift_base_s = float("inf")
        for _ in range(5):
            lift_base_s = min(lift_base_s, _best_of(1, _lift_baseline, df, key, "sku"))
            lift_s = min(lift_s, _best_of(1, templates.basket_lift, df, key, "sku"))
        lift_ratio = lift_s / lift_base_s
        lines.append(f"OUTCOME-BUDGET retail basket_lift ratio={lift_ratio:.3f} k={BASKET_LIFT_BUDGET_RATIO} "
                     f"lift_s={lift_s:.3f} baseline_s={lift_base_s:.3f}")
    request.config.stash.setdefault(BUDGET_LINES, []).extend(lines)
    print("\n" + "\n".join(lines))

    assert ratio <= BUILD_BUDGET_RATIO, (
        f"{which}: the 200k-row build took {ratio:.2f}x the vectorised baseline "
        f"({build_s:.3f} s vs {baseline_s:.3f} s); budget {BUILD_BUDGET_RATIO}x")
    if which == "retail":
        assert lift_ratio <= BASKET_LIFT_BUDGET_RATIO, (
            f"basket_lift took {lift_ratio:.2f}x the same work done vectorised "
            f"({lift_s:.3f} s vs {lift_base_s:.3f} s); budget {BASKET_LIFT_BUDGET_RATIO}x")
