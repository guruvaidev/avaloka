"""Complex data-manipulation task corpus with INDEPENDENT ground truth.

Every task here exercises an operation family that the simple-prompt path does
not reach: joins, window functions, pivots/unpivots, time series, groupby
-transform, and multi-step chains.

The acceptance checks in this module are the *measurement* ground truth. They
are written by hand and are deliberately independent of anything the agent
pipeline emits -- in particular they are NOT the blueprint contract the coder
produces. Keeping them separate is what makes a before/after number honest:
if the pipeline's own contract were the yardstick, the pipeline could pass by
lowering its own bar.

Each check returns a list of (name, passed, detail). A task counts as a success
only when every check passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DATA = REPO_ROOT / "app" / "sample_data"
KAGGLE_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "mta_kaggle"

Check = Tuple[str, bool, str]
CheckFn = Callable[[pd.DataFrame, pd.DataFrame], List[Check]]


# ---------------------------------------------------------------------------
# check helpers
# ---------------------------------------------------------------------------

def _norm(name: Any) -> str:
    return str(name).strip().lower().replace(" ", "_")


def has_col_like(df: pd.DataFrame, *candidates: str) -> str | None:
    """Find a column whose normalised name matches or contains a candidate.

    Generated code names columns freely ("avg_billing", "mean_billing_amount"),
    so an exact-name check would measure naming luck rather than correctness.
    """
    norm = {_norm(c): c for c in df.columns}
    for cand in candidates:
        c = _norm(cand)
        if c in norm:
            return norm[c]
    for cand in candidates:
        c = _norm(cand)
        for n, original in norm.items():
            if c in n or n in c:
                return original
    return None


def is_numeric(df: pd.DataFrame, col: str) -> bool:
    return pd.api.types.is_numeric_dtype(df[col])


def check_is_aggregate(result: pd.DataFrame, source: pd.DataFrame) -> Check:
    """The single most-violated postcondition: an aggregation must not return
    the input rows back. This is the deterministic form of the coder prompt's
    'AGGREGATION / CHART OUTPUT RULE'."""
    shrank = len(result) < len(source)
    return (
        "result is aggregated (fewer rows than input)",
        shrank,
        f"{len(result)} result rows vs {len(source)} input rows",
    )


def check_one_row_per_group(result: pd.DataFrame, group_cols: List[str]) -> Check:
    present = [c for c in group_cols if c in result.columns]
    if len(present) != len(group_cols):
        return ("one row per group", False, f"group columns missing: {group_cols}")
    dupes = int(result.duplicated(subset=present).sum())
    return ("one row per group (keys unique)", dupes == 0, f"{dupes} duplicate key rows")


def check_rate_bounds(result: pd.DataFrame, col: str) -> Check:
    if col not in result.columns or not is_numeric(result, col):
        return (f"{col} is a numeric rate", False, "missing or non-numeric")
    s = result[col].dropna()
    ok = bool(len(s)) and float(s.min()) >= -1e-9 and float(s.max()) <= 1.0 + 1e-9
    return (f"{col} within [0,1]", ok, f"min={s.min() if len(s) else 'NA'} max={s.max() if len(s) else 'NA'}")


# ---------------------------------------------------------------------------
# task definition
# ---------------------------------------------------------------------------

@dataclass
class ComplexTask:
    id: str
    family: str          # join | window | pivot | timeseries | groupby_transform | chain
    prompt: str
    dataset: Path
    check: CheckFn
    # Extra datasets the prompt refers to (multi-dataset / join tasks).
    extra_datasets: Dict[str, Path] = field(default_factory=dict)
    notes: str = ""

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.dataset)


# ---------------------------------------------------------------------------
# healthcare  (55,500 rows; 'Date of Admission' + 'Discharge Date' are real dates)
# ---------------------------------------------------------------------------

HEALTHCARE = SAMPLE_DATA / "healthcare_dataset.csv"


def _chk_los_by_condition(result: pd.DataFrame, source: pd.DataFrame) -> List[Check]:
    """Mean length-of-stay per medical condition: date arithmetic + groupby."""
    out: List[Check] = [check_is_aggregate(result, source)]
    cond = has_col_like(result, "Medical Condition", "condition")
    los = has_col_like(result, "length_of_stay", "los", "stay_days", "days", "duration")
    out.append(("has condition key column", cond is not None, str(list(result.columns))))
    out.append(("has length-of-stay metric", los is not None, str(list(result.columns))))
    if cond:
        out.append(check_one_row_per_group(result, [cond]))
        n_expected = source["Medical Condition"].nunique()
        out.append((
            "one row per distinct condition",
            len(result) == n_expected,
            f"{len(result)} rows vs {n_expected} distinct conditions",
        ))
    if los and is_numeric(result, los):
        s = result[los].dropna()
        # Ground truth from the raw data.
        src = source.copy()
        truth = (
            pd.to_datetime(src["Discharge Date"], errors="coerce")
            - pd.to_datetime(src["Date of Admission"], errors="coerce")
        ).dt.days
        lo, hi = float(truth.min()), float(truth.max())
        ok = bool(len(s)) and float(s.min()) >= lo - 1 and float(s.max()) <= hi + 1
        out.append((
            "stay values within true per-row range",
            ok,
            f"result [{s.min() if len(s) else 'NA'},{s.max() if len(s) else 'NA'}] vs true [{lo},{hi}]",
        ))
    return out


def _chk_abnormal_rate_by_hospital(result: pd.DataFrame, source: pd.DataFrame) -> List[Check]:
    """Rate of Abnormal test results per hospital: 0/1 mean over a group."""
    out: List[Check] = [check_is_aggregate(result, source)]
    hosp = has_col_like(result, "Hospital")
    rate = has_col_like(result, "abnormal_rate", "rate", "share", "pct", "percentage", "proportion")
    out.append(("has hospital key", hosp is not None, str(list(result.columns))))
    out.append(("has rate metric", rate is not None, str(list(result.columns))))
    if hosp:
        out.append(check_one_row_per_group(result, [hosp]))
    if rate and is_numeric(result, rate):
        s = result[rate].dropna()
        # Accept either a 0-1 proportion or a 0-100 percentage.
        as_pct = float(s.max()) > 1.0 + 1e-9
        scaled = s / 100.0 if as_pct else s
        ok = bool(len(scaled)) and scaled.min() >= -1e-9 and scaled.max() <= 1.0 + 1e-9
        out.append(("rate within [0,1] (or [0,100])", ok, f"min={s.min()} max={s.max()}"))
    return out


def _chk_billing_share_within_insurer(result: pd.DataFrame, source: pd.DataFrame) -> List[Check]:
    """groupby-transform: each condition's share of its insurer's total billing.

    The share column must sum to ~1 within each insurer -- a postcondition a
    row-wise mistake cannot satisfy."""
    out: List[Check] = [check_is_aggregate(result, source)]
    ins = has_col_like(result, "Insurance Provider", "insurer", "provider")
    cond = has_col_like(result, "Medical Condition", "condition")
    share = has_col_like(result, "share", "pct", "percentage", "proportion", "ratio")
    out.append(("has insurer key", ins is not None, str(list(result.columns))))
    out.append(("has condition key", cond is not None, str(list(result.columns))))
    out.append(("has share metric", share is not None, str(list(result.columns))))
    if ins and cond:
        out.append(check_one_row_per_group(result, [ins, cond]))
    if ins and share and is_numeric(result, share):
        sums = result.groupby(ins)[share].sum()
        as_pct = float(sums.median()) > 1.5
        target = 100.0 if as_pct else 1.0
        ok = bool(len(sums)) and bool(((sums - target).abs() < 0.02 * target).all())
        out.append((
            "shares sum to 1 within each insurer",
            ok,
            f"per-insurer sums range [{sums.min():.4f},{sums.max():.4f}], target {target}",
        ))
    return out


def _chk_monthly_admissions_trend(result: pd.DataFrame, source: pd.DataFrame) -> List[Check]:
    """Time series: monthly admission counts with a 3-month rolling average."""
    out: List[Check] = [check_is_aggregate(result, source)]
    period = has_col_like(result, "month", "period", "year_month", "date")
    count = has_col_like(result, "admissions", "count", "n", "total")
    roll = has_col_like(result, "rolling", "moving", "ma", "roll3", "rolling_avg")
    out.append(("has month/period column", period is not None, str(list(result.columns))))
    out.append(("has admission count", count is not None, str(list(result.columns))))
    out.append(("has rolling average column", roll is not None, str(list(result.columns))))
    truth_months = (
        pd.to_datetime(source["Date of Admission"], errors="coerce")
        .dt.to_period("M").nunique()
    )
    out.append((
        "one row per month present in data",
        len(result) == truth_months,
        f"{len(result)} rows vs {truth_months} distinct months",
    ))
    if count and is_numeric(result, count):
        total = float(result[count].sum())
        out.append((
            "monthly counts sum to total rows",
            abs(total - len(source)) < 1,
            f"sum={total} vs {len(source)} input rows",
        ))
    return out


def _chk_top_doctor_per_hospital(result: pd.DataFrame, source: pd.DataFrame) -> List[Check]:
    """Window/rank: the highest-billing doctor within each hospital."""
    out: List[Check] = [check_is_aggregate(result, source)]
    hosp = has_col_like(result, "Hospital")
    doc = has_col_like(result, "Doctor")
    out.append(("has hospital column", hosp is not None, str(list(result.columns))))
    out.append(("has doctor column", doc is not None, str(list(result.columns))))
    if hosp:
        n_hosp = source["Hospital"].nunique()
        out.append(check_one_row_per_group(result, [hosp]))
        out.append((
            "exactly one row per hospital",
            len(result) == n_hosp,
            f"{len(result)} rows vs {n_hosp} hospitals",
        ))
    if hosp and doc:
        # Every (hospital, doctor) pair returned must actually exist in the data.
        pairs = set(zip(source["Hospital"], source["Doctor"]))
        bad = [
            (h, d) for h, d in zip(result[hosp], result[doc])
            if (h, d) not in pairs
        ]
        out.append((
            "every hospital/doctor pair exists in source",
            not bad,
            f"{len(bad)} invented pairs (e.g. {bad[:2]})",
        ))
    return out


def _chk_age_band_condition_pivot(result: pd.DataFrame, source: pd.DataFrame) -> List[Check]:
    """Pivot: age bands as rows, medical conditions as columns, counts as values."""
    out: List[Check] = [check_is_aggregate(result, source)]
    conditions = set(_norm(c) for c in source["Medical Condition"].unique())
    cols = set(_norm(c) for c in result.columns)
    matched = conditions & cols
    out.append((
        "medical conditions became columns",
        len(matched) >= max(2, len(conditions) - 1),
        f"{len(matched)}/{len(conditions)} condition columns found in {list(result.columns)}",
    ))
    out.append((
        "few rows (age bands, not raw rows)",
        1 < len(result) <= 30,
        f"{len(result)} rows",
    ))
    numeric_cols = [c for c in result.columns if is_numeric(result, c)]
    if numeric_cols:
        total = float(result[numeric_cols].sum().sum())
        out.append((
            "cell counts sum to input row count",
            abs(total - len(source)) <= len(source) * 0.01,
            f"cells sum to {total} vs {len(source)} input rows",
        ))
    return out


# ---------------------------------------------------------------------------
# taxi  (50,426 rows; real pickup/dropoff datetimes)
# ---------------------------------------------------------------------------

TAXI = SAMPLE_DATA / "yellow_tripdata_2015-01_dataset_50k.csv"


def _chk_hourly_tip_rate(result: pd.DataFrame, source: pd.DataFrame) -> List[Check]:
    """Time series + derived rate: average tip percentage by pickup hour."""
    out: List[Check] = [check_is_aggregate(result, source)]
    hour = has_col_like(result, "hour", "pickup_hour")
    tip = has_col_like(result, "tip_pct", "tip_percentage", "tip_rate", "tip_share", "avg_tip")
    out.append(("has hour column", hour is not None, str(list(result.columns))))
    out.append(("has tip rate column", tip is not None, str(list(result.columns))))
    out.append((
        "at most 24 rows (one per hour)",
        1 < len(result) <= 24,
        f"{len(result)} rows",
    ))
    if hour and is_numeric(result, hour):
        s = result[hour].dropna()
        ok = bool(len(s)) and s.min() >= 0 and s.max() <= 23
        out.append(("hour values in 0..23", ok, f"min={s.min() if len(s) else 'NA'} max={s.max() if len(s) else 'NA'}"))
    return out


def _chk_trip_duration_by_distance_band(result: pd.DataFrame, source: pd.DataFrame) -> List[Check]:
    """Multi-step chain: datetime diff -> duration, qcut distance bands, aggregate."""
    out: List[Check] = [check_is_aggregate(result, source)]
    band = has_col_like(result, "band", "bin", "bucket", "quartile", "decile", "distance_group", "distance")
    dur = has_col_like(result, "duration", "minutes", "trip_time", "avg_duration", "mean_duration")
    out.append(("has distance band column", band is not None, str(list(result.columns))))
    out.append(("has duration metric", dur is not None, str(list(result.columns))))
    out.append(("small number of bands", 1 < len(result) <= 20, f"{len(result)} rows"))
    if dur and is_numeric(result, dur):
        s = result[dur].dropna()
        out.append((
            "durations positive",
            bool(len(s)) and float(s.min()) > 0,
            f"min={s.min() if len(s) else 'NA'}",
        ))
    return out


def _chk_daily_revenue_running_total(result: pd.DataFrame, source: pd.DataFrame) -> List[Check]:
    """Window function: daily revenue with a cumulative running total."""
    out: List[Check] = [check_is_aggregate(result, source)]
    day = has_col_like(result, "date", "day", "pickup_date")
    daily = has_col_like(result, "daily_revenue", "revenue", "total_amount", "daily_total")
    cum = has_col_like(result, "cumulative", "running", "cumsum", "running_total")
    out.append(("has date column", day is not None, str(list(result.columns))))
    out.append(("has daily revenue", daily is not None, str(list(result.columns))))
    out.append(("has running total", cum is not None, str(list(result.columns))))
    if cum and is_numeric(result, cum):
        s = result[cum].dropna()
        ok = bool(len(s)) and bool((s.diff().dropna() >= -1e-6).all())
        out.append(("running total is non-decreasing", ok, "monotonicity of cumulative column"))
        if daily and is_numeric(result, daily):
            ok2 = abs(float(s.iloc[-1]) - float(result[daily].sum())) < max(1.0, 0.001 * float(result[daily].sum()))
            out.append((
                "final running total equals sum of daily",
                ok2,
                f"last cum={s.iloc[-1]} vs sum daily={result[daily].sum()}",
            ))
    return out


# ---------------------------------------------------------------------------
# instacart  (join / co-occurrence over committed Kaggle fixtures)
# ---------------------------------------------------------------------------

INSTACART_PRIOR = KAGGLE_FIXTURES / "instacart_prior_sample.csv"
INSTACART_TRAIN = KAGGLE_FIXTURES / "instacart_train_sample.csv"


def _chk_product_reorder_join(result: pd.DataFrame, source: pd.DataFrame) -> List[Check]:
    """Join two order tables on product_id and compare reorder behaviour."""
    out: List[Check] = [check_is_aggregate(result, source)]
    prod = has_col_like(result, "product_id")
    out.append(("has product_id", prod is not None, str(list(result.columns))))
    if prod:
        out.append(check_one_row_per_group(result, [prod]))
        src_products = set(source["product_id"].unique())
        got = set(result[prod].dropna().unique())
        out.append((
            "no invented product ids",
            got.issubset(src_products),
            f"{len(got - src_products)} product ids not in the prior table",
        ))
    rate_cols = [c for c in result.columns if is_numeric(result, c) and "rate" in _norm(c)]
    for c in rate_cols:
        out.append(check_rate_bounds(result, c))
    return out


# ---------------------------------------------------------------------------
# walmart  (wide -> long unpivot + window)
# ---------------------------------------------------------------------------

WALMART = KAGGLE_FIXTURES / "walmart_validation_sample.csv"


def _chk_walmart_melt_window(result: pd.DataFrame, source: pd.DataFrame) -> List[Check]:
    """Unpivot d_* day columns to long form, then aggregate per store per day."""
    out: List[Check] = []
    day_cols = [c for c in source.columns if str(c).startswith("d_")]
    store = has_col_like(result, "store_id")
    day = has_col_like(result, "day", "d", "date", "day_num")
    units = has_col_like(result, "units", "sales", "total", "sum", "quantity")
    out.append(("has store_id", store is not None, str(list(result.columns))))
    out.append(("has day column", day is not None, str(list(result.columns))))
    out.append(("has units metric", units is not None, str(list(result.columns))))
    out.append((
        "wide day columns were unpivoted away",
        not any(str(c).startswith("d_") and c in result.columns for c in day_cols),
        f"still-wide columns: {[c for c in day_cols if c in result.columns]}",
    ))
    if store and day:
        out.append(check_one_row_per_group(result, [store, day]))
        n_expected = source["store_id"].nunique() * len(day_cols)
        out.append((
            "one row per store per day",
            len(result) == n_expected,
            f"{len(result)} rows vs {n_expected} store x day combinations",
        ))
    if units and is_numeric(result, units):
        total = float(result[units].sum())
        truth = float(source[day_cols].sum().sum())
        out.append((
            "unit totals preserved through the unpivot",
            abs(total - truth) < max(1.0, truth * 0.001),
            f"result total {total} vs source total {truth}",
        ))
    return out


# ---------------------------------------------------------------------------
# the corpus
# ---------------------------------------------------------------------------

TASKS: List[ComplexTask] = [
    ComplexTask(
        id="hc_los_by_condition",
        family="chain",
        prompt=(
            "For each medical condition, compute the average length of stay in days "
            "(discharge date minus admission date) and the number of admissions. "
            "Return one row per condition sorted by average stay descending."
        ),
        dataset=HEALTHCARE,
        check=_chk_los_by_condition,
        notes="date arithmetic + groupby aggregate",
    ),
    ComplexTask(
        id="hc_abnormal_rate_by_hospital",
        family="groupby_transform",
        prompt=(
            "Calculate the proportion of test results that are 'Abnormal' for each "
            "hospital, along with the number of records per hospital. Return one row "
            "per hospital."
        ),
        dataset=HEALTHCARE,
        check=_chk_abnormal_rate_by_hospital,
        notes="0/1 rate as a grouped mean",
    ),
    ComplexTask(
        id="hc_billing_share_within_insurer",
        family="groupby_transform",
        prompt=(
            "For each insurance provider, show how total billing amount is split "
            "across medical conditions: give the billing total for each "
            "provider/condition pair and that pair's share of the provider's overall "
            "billing."
        ),
        dataset=HEALTHCARE,
        check=_chk_billing_share_within_insurer,
        notes="groupby-transform share-of-total; shares must sum to 1 per group",
    ),
    ComplexTask(
        id="hc_monthly_admissions_rolling",
        family="timeseries",
        prompt=(
            "Build a monthly time series of admission counts from the date of "
            "admission, and add a 3-month rolling average of that count."
        ),
        dataset=HEALTHCARE,
        check=_chk_monthly_admissions_trend,
        notes="resample to month + rolling window",
    ),
    ComplexTask(
        id="hc_top_doctor_per_hospital",
        family="window",
        prompt=(
            "For every hospital, identify the doctor with the highest total billing "
            "amount and report that doctor's total. Return exactly one row per hospital."
        ),
        dataset=HEALTHCARE,
        check=_chk_top_doctor_per_hospital,
        notes="rank/idxmax within group (argmax window)",
    ),
    ComplexTask(
        id="hc_age_band_condition_pivot",
        family="pivot",
        prompt=(
            "Create a pivot table with age bands of 10 years as rows and medical "
            "condition as columns, containing the number of patients in each cell."
        ),
        dataset=HEALTHCARE,
        check=_chk_age_band_condition_pivot,
        notes="binning + pivot",
    ),
    ComplexTask(
        id="taxi_hourly_tip_rate",
        family="timeseries",
        prompt=(
            "For each hour of the day (from the pickup timestamp), compute the average "
            "tip as a percentage of the fare amount, and the number of trips. Return "
            "one row per hour."
        ),
        dataset=TAXI,
        check=_chk_hourly_tip_rate,
        notes="datetime extraction + derived ratio + groupby",
    ),
    ComplexTask(
        id="taxi_duration_by_distance_band",
        family="chain",
        prompt=(
            "Compute each trip's duration in minutes from the pickup and dropoff "
            "timestamps, split trips into 5 distance quintiles, and report the average "
            "duration and average total amount for each quintile."
        ),
        dataset=TAXI,
        check=_chk_trip_duration_by_distance_band,
        notes="multi-step chain: datetime diff -> qcut -> aggregate",
    ),
    ComplexTask(
        id="taxi_daily_revenue_running_total",
        family="window",
        prompt=(
            "Produce a daily series of total revenue (total amount) from the pickup "
            "date, and add a cumulative running total of revenue across days."
        ),
        dataset=TAXI,
        check=_chk_daily_revenue_running_total,
        notes="resample + cumsum window",
    ),
    ComplexTask(
        id="instacart_product_reorder",
        family="join",
        prompt=(
            "For each product, compute how many times it was ordered and its reorder "
            "rate. Return one row per product ordered by reorder rate descending."
        ),
        dataset=INSTACART_PRIOR,
        extra_datasets={"train": INSTACART_TRAIN},
        check=_chk_product_reorder_join,
        notes="groupby aggregate with a bounded rate",
    ),
    ComplexTask(
        id="walmart_melt_store_day",
        family="pivot",
        prompt=(
            "The daily sales are stored in wide form, one column per day (the d_ "
            "columns). Reshape the data into long form and give total units sold per "
            "store per day."
        ),
        dataset=WALMART,
        check=_chk_walmart_melt_window,
        notes="unpivot (melt) + regroup; totals must be preserved",
    ),
]

TASKS_BY_ID = {t.id: t for t in TASKS}


def families() -> Dict[str, List[ComplexTask]]:
    out: Dict[str, List[ComplexTask]] = {}
    for t in TASKS:
        out.setdefault(t.family, []).append(t)
    return out
