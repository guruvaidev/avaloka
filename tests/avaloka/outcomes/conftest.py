"""Fixtures for the outcome tests: a retail and a fintech table, seeded.

``python tests/avaloka/outcomes/conftest.py`` regenerates the browser fixture
``ui/tests/e2e/fixtures/outcome-export.html``: the analyst export of the report
built from :func:`retail_frame`. The golden examples cannot serve there: their
chart options are abridged (a bar series with no axis), which ECharts rejects.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

#: Lines the 200k-row budget test records on every run, pass or fail. CI runs
#: ``pytest -q`` without ``-s`` or ``-rA``, so a print would show only on failure;
#: the terminal summary below shows the measured ratios and k on every run, and
#: only when that test ran.
BUDGET_LINES = pytest.StashKey[list]()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    lines = config.stash.get(BUDGET_LINES, None)
    if lines:
        terminalreporter.write_sep("-", "outcomes 200k-row budget (ratios to baselines timed in the same run)")
        for line in lines:
            terminalreporter.write_line(line)


GOLDEN =Path(__file__).resolve().parents[3] / "avaloka" / "outcomes" / "golden"
CREATED = "2026-10-18T16:02:11Z"
ANALYSIS = {"analysis_id": "a-77", "thread_id": "t-12", "turn_id": "turn-3", "version": 1, "dataset_ids": ["ds-sales"]}

# Orders per state. WY and VT get one line per order, so 4 and 9 rows: under k = 10.
STATE_ROWS = {"CA": 160, "NY": 120, "TX": 110, "WA": 60, "IL": 40, "WY": 4, "VT": 9}
SMALL = {"WY", "VT"}


def retail_frame(seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    products = [f"sku_{c}" for c in "abcdefgh"]
    cats = {"sku_a": "toys", "sku_b": "toys", "sku_c": "books", "sku_d": "books",
            "sku_e": "garden", "sku_f": "garden", "sku_g": "kitchen", "sku_h": "kitchen"}
    order = 0
    for state, n in STATE_ROWS.items():
        for _ in range(n):
            order += 1
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 730)))
            items = rng.choice(products, size=1 if state in SMALL else int(rng.integers(1, 4)), replace=False)
            if state not in SMALL and rng.random() < 0.5:
                items = np.unique(np.append(items, ["sku_a", "sku_c"]))
            for p in items:
                base = 10 + products.index(p) * 3
                price = round(base * (1.1 if date.year == 2025 else 1.0) * rng.uniform(0.95, 1.05), 2)
                q = int(rng.integers(1, 6))
                rows.append({"order_id": f"o{order:05d}", "order_date": date, "state": state, "sku": p,
                             "category": cats[p], "unit_price": price, "quantity": q,
                             "units_received": q + int(rng.integers(0, 3))})
    return pd.DataFrame(rows)


def fintech_frame(seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    merchants = [f"m{i:02d}" for i in range(40)]
    for acct in range(600):
        orig = pd.Timestamp("2025-01-01") + pd.DateOffset(months=int(rng.integers(0, 9)))
        dpd = 0
        merchant = rng.choice(merchants, p=np.r_[[0.2, 0.1], np.full(38, 0.7 / 38)])
        bal = float(rng.uniform(500, 5000))
        for k in range(1, 7):
            obs = orig + pd.DateOffset(months=k)
            if obs > pd.Timestamp("2026-06-01"):
                break
            dpd = 0 if rng.random() < 0.85 else min(120, dpd + 30)
            rows.append({"account_id": f"A{acct:05d}", "origination_date": orig, "as_of_date": obs,
                         "days_past_due": dpd, "balance": round(bal, 2), "merchant_name": merchant,
                         "interest_income": round(bal * 0.01, 2), "charge_offs": -round(bal if dpd >= 120 else 0.0, 2),
                         "currency": "USD"})
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def retail_df():
    return retail_frame()


@pytest.fixture(scope="session")
def fintech_df():
    return fintech_frame()


@pytest.fixture(scope="session")
def golden_retail():
    return json.loads((GOLDEN / "example_retail.json").read_text())


@pytest.fixture(scope="session")
def golden_fintech():
    return json.loads((GOLDEN / "example_fintech.json").read_text())


E2E_FIXTURE = Path(__file__).resolve().parents[3] / "ui" / "tests" / "e2e" / "fixtures" / "outcome-export.html"


def with_linear_whatif(report: dict) -> dict:
    """``report`` plus one linear what-if on net sales (the verify branch will
    produce these; 2.0's runtime computes ``linear`` and ``step`` only)."""
    from avaloka.outcomes import spec
    r = json.loads(json.dumps(report))
    figs = {f["id"]: f for f in r["figures"]}
    y0 = figs[next(k["value"] for k in r["kpis"] if k["id"] == "net_sales")]["value"]
    r["whatif"] = [{"id": "w1", "kpi": "net_sales", "y0": y0, "z": 1.96, "evidence_kind": "associational",
                    "drivers": [{"id": "discount", "label": "Average discount", "unit": "percent",
                                 "baseline": 10, "lo": 5, "hi": 25, "step": 0.5, "form": "linear",
                                 "beta": round(y0 * 0.004, 2), "se": round(y0 * 0.001, 2)}],
                    "evidence": {"turn_id": "turn-3"}}]
    spec.assert_valid(r)
    return r


def e2e_export() -> str:
    from avaloka.outcomes.render_html import render_html
    from avaloka.outcomes.templates import build_report
    report = build_report(retail_frame(), "how are sales doing", analysis=ANALYSIS, created_at=CREATED)
    return render_html(with_linear_whatif(report), audience="analyst")


if __name__ == "__main__":
    E2E_FIXTURE.write_text(e2e_export(), encoding="utf-8")
    print(f"wrote {E2E_FIXTURE} ({E2E_FIXTURE.stat().st_size:,} bytes)")
