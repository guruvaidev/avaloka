#!/usr/bin/env python3
"""Synthetic customer-churn dataset for an Avaloka demo. Offline, reproducible, no real people."""
import argparse
import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_id = np.array([f"CUST-{i:06d}" for i in range(1, rows + 1)])

    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=rows, p=[0.55, 0.27, 0.18],
    )
    payment_method = rng.choice(
        ["Credit card", "Bank transfer", "Electronic check", "Mailed check"],
        size=rows, p=[0.34, 0.25, 0.29, 0.12],
    )
    region = rng.choice(
        ["North", "South", "East", "West", "Central"],
        size=rows, p=[0.22, 0.24, 0.19, 0.20, 0.15],
    )
    signup_channel = rng.choice(
        ["Organic", "Paid search", "Referral", "Partner", "Outbound"],
        size=rows, p=[0.30, 0.24, 0.20, 0.14, 0.12],
    )
    has_premium_support = rng.random(rows) < 0.28

    tenure_months = rng.integers(1, 73, size=rows)
    monthly_charges = np.round(rng.normal(70, 25, size=rows), 2)
    monthly_charges = np.clip(monthly_charges, 15, 160)
    total_charges = np.round(
        monthly_charges * tenure_months * rng.normal(1.0, 0.05, size=rows), 2
    )
    num_support_tickets = rng.poisson(1.6, size=rows)
    data_usage_gb = np.round(np.abs(rng.normal(240, 130, size=rows)), 1)
    last_login_days_ago = rng.integers(0, 90, size=rows)

    contract_effect = np.select(
        [contract_type == "Month-to-month",
         contract_type == "One year",
         contract_type == "Two year"],
        [1.15, -0.35, -1.05],
    )
    channel_effect = np.select(
        [signup_channel == "Outbound", signup_channel == "Referral"],
        [0.45, -0.30], default=0.0,
    )
    logit = (
        -1.35
        + contract_effect
        + channel_effect
        - 0.030 * tenure_months
        + 0.180 * num_support_tickets
        + 0.020 * last_login_days_ago
        - 0.55 * has_premium_support.astype(float)
        + rng.normal(0, 0.45, size=rows)
    )
    churned = (rng.random(rows) < sigmoid(logit)).astype(int)

    account_status = np.where(churned == 1, "closed", "active")
    flip = rng.random(rows) < 0.005
    account_status = np.where(flip,
                              np.where(account_status == "closed", "active", "closed"),
                              account_status)

    avg_ticket_resolution_days = np.round(
        np.abs(rng.normal(3.5, 1.8, size=rows)), 2
    )
    avg_ticket_resolution_days = np.where(
        num_support_tickets == 0, np.nan, avg_ticket_resolution_days
    )
    extra_missing = rng.random(rows) < 0.03
    avg_ticket_resolution_days = np.where(
        extra_missing, np.nan, avg_ticket_resolution_days
    )
    usage_missing = rng.random(rows) < 0.02
    data_usage_gb = np.where(usage_missing, np.nan, data_usage_gb)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "tenure_months": tenure_months,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "num_support_tickets": num_support_tickets,
        "avg_ticket_resolution_days": avg_ticket_resolution_days,
        "has_premium_support": has_premium_support,
        "data_usage_gb": data_usage_gb,
        "region": region,
        "signup_channel": signup_channel,
        "last_login_days_ago": last_login_days_ago,
        "account_status": account_status,
        "churned": churned,
    })

    dirty = rng.choice(rows, size=max(8, rows // 800), replace=False)
    df.loc[dirty[: len(dirty) // 2], "monthly_charges"] = -1 * df.loc[
        dirty[: len(dirty) // 2], "monthly_charges"
    ]
    df.loc[dirty[len(dirty) // 2:], "total_charges"] = 0.0

    return df


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic churn dataset.")
    ap.add_argument("--rows", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", default="customer_churn")
    args = ap.parse_args()

    df = build(args.rows, args.seed)

    with_trap = f"{args.prefix}.csv"
    clean = f"{args.prefix}_clean.csv"
    df.to_csv(with_trap, index=False)
    df.drop(columns=["account_status"]).to_csv(clean, index=False)

    rate = df["churned"].mean()
    print(f"Wrote {with_trap}  ({len(df):,} rows, {df.shape[1]} cols) - WITH leakage trap 'account_status'")
    print(f"Wrote {clean}  ({len(df):,} rows, {df.shape[1]-1} cols) - trap removed")
    print(f"Churn rate: {rate:.1%}  (majority-class baseline accuracy ~ {max(rate,1-rate):.1%})")
    print(f"Missing avg_ticket_resolution_days: {df['avg_ticket_resolution_days'].isna().mean():.1%}")
    print("corr(account_status=='closed', churned) =",
          round(np.corrcoef((df['account_status'] == 'closed').astype(int), df['churned'])[0, 1], 4))


if __name__ == "__main__":
    main()
