"""Hand-written correct solutions for every task in ``tasks.py``.

These exist to keep the measurement honest. Acceptance criteria have to be
falsifiable in both directions:

  * they must REJECT the untransformed input frame (otherwise they measure
    nothing), and
  * they must be SATISFIABLE by correct code (otherwise a "before" score of
    zero says nothing about the pipeline).

``test_harness.py`` asserts both. These solutions are the second half -- they
are not used by the pipeline and are never shown to a model.
"""

from __future__ import annotations

from typing import Dict

REFERENCE_SOLUTIONS: Dict[str, str] = {}

REFERENCE_SOLUTIONS["hc_los_by_condition"] = '''
def main(df):
    d = df.copy()
    d["length_of_stay"] = (pd.to_datetime(d["Discharge Date"]) - pd.to_datetime(d["Date of Admission"])).dt.days
    g = d.groupby("Medical Condition", as_index=False).agg(
        length_of_stay=("length_of_stay","mean"), admissions=("Name","count"))
    return g.sort_values("length_of_stay", ascending=False)
'''

REFERENCE_SOLUTIONS["hc_abnormal_rate_by_hospital"] = '''
def main(df):
    d = df.copy()
    d["is_abnormal"] = (d["Test Results"] == "Abnormal").astype(int)
    return d.groupby("Hospital", as_index=False).agg(
        abnormal_rate=("is_abnormal","mean"), records=("Name","count"))
'''

REFERENCE_SOLUTIONS["hc_billing_share_within_insurer"] = '''
def main(df):
    g = df.groupby(["Insurance Provider","Medical Condition"], as_index=False)["Billing Amount"].sum()
    g["provider_total"] = g.groupby("Insurance Provider")["Billing Amount"].transform("sum")
    g["share"] = g["Billing Amount"] / g["provider_total"]
    return g
'''

REFERENCE_SOLUTIONS["hc_monthly_admissions_rolling"] = '''
def main(df):
    d = df.copy()
    d["month"] = pd.to_datetime(d["Date of Admission"]).dt.to_period("M").astype(str)
    g = d.groupby("month", as_index=False).agg(admissions=("Name","count")).sort_values("month")
    g["rolling_avg"] = g["admissions"].rolling(3, min_periods=1).mean()
    return g
'''

REFERENCE_SOLUTIONS["hc_top_doctor_per_hospital"] = '''
def main(df):
    g = df.groupby(["Hospital","Doctor"], as_index=False)["Billing Amount"].sum()
    idx = g.groupby("Hospital")["Billing Amount"].idxmax()
    return g.loc[idx].reset_index(drop=True)
'''

REFERENCE_SOLUTIONS["hc_age_band_condition_pivot"] = '''
def main(df):
    d = df.copy()
    d["age_band"] = (d["Age"] // 10 * 10).astype(int).astype(str) + "-" + ((d["Age"] // 10 * 10) + 9).astype(int).astype(str)
    p = pd.pivot_table(d, index="age_band", columns="Medical Condition",
                       values="Name", aggfunc="count", fill_value=0)
    return p.reset_index()
'''

REFERENCE_SOLUTIONS["taxi_hourly_tip_rate"] = '''
def main(df):
    d = df.copy()
    d["hour"] = pd.to_datetime(d["tpep_pickup_datetime"]).dt.hour
    d = d[d["fare_amount"] > 0]
    d["tip_pct"] = d["tip_amount"] / d["fare_amount"]
    return d.groupby("hour", as_index=False).agg(tip_pct=("tip_pct","mean"), trips=("VendorID","count"))
'''

REFERENCE_SOLUTIONS["taxi_duration_by_distance_band"] = '''
def main(df):
    d = df.copy()
    d["duration_minutes"] = (pd.to_datetime(d["tpep_dropoff_datetime"]) - pd.to_datetime(d["tpep_pickup_datetime"])).dt.total_seconds()/60
    d = d[d["duration_minutes"] > 0]
    d["distance_band"] = pd.qcut(d["trip_distance"], 5, labels=False, duplicates="drop")
    return d.groupby("distance_band", as_index=False).agg(
        duration_minutes=("duration_minutes","mean"), avg_total=("total_amount","mean"))
'''

REFERENCE_SOLUTIONS["taxi_daily_revenue_running_total"] = '''
def main(df):
    d = df.copy()
    d["date"] = pd.to_datetime(d["tpep_pickup_datetime"]).dt.date.astype(str)
    g = d.groupby("date", as_index=False).agg(daily_revenue=("total_amount","sum")).sort_values("date")
    g["running_total"] = g["daily_revenue"].cumsum()
    return g
'''

REFERENCE_SOLUTIONS["instacart_product_reorder"] = '''
def main(df):
    g = df.groupby("product_id", as_index=False).agg(
        times_ordered=("order_id","count"), reorder_rate=("reordered","mean"))
    return g.sort_values("reorder_rate", ascending=False)
'''

REFERENCE_SOLUTIONS["walmart_melt_store_day"] = '''
def main(df):
    day_cols = [c for c in df.columns if str(c).startswith("d_")]
    lg = df.melt(id_vars=["store_id"], value_vars=day_cols, var_name="day", value_name="units")
    return lg.groupby(["store_id","day"], as_index=False)["units"].sum()
'''
