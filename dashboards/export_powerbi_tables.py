"""
Builds the flat, Power-BI-ready export tables backing the 8-page
Fraud Risk Command Center:

  1. Executive Overview     -> fact_daily_summary.csv
  2. Fraud Trends           -> fact_daily_summary.csv (same table, different visuals)
  3. Transaction Risk       -> fact_transactions_scored.csv
  4. Customer Risk          -> dim_customer_risk.csv
  5. Merchant Analysis      -> dim_merchant_risk.csv
  6. Device & Geographic    -> dim_device_risk.csv, fact_geographic_summary.csv
  7. Investigation Queue    -> fact_investigation_queue.csv
  8. Model Performance      -> fact_model_performance.csv (from model_comparison.json
                                + policy_evaluation.json)

Power BI connects directly to these CSVs (or the equivalent parquet files);
relationships and DAX measures are documented in dax_measures.md alongside
this script rather than embedded in a .pbix, since a .pbix can't be authored
headlessly here.
"""

import os
import json
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
MODELS_DIR = os.path.join(REPO_ROOT, "src", "models", "artifacts")
OUT_DIR = os.path.join(REPO_ROOT, "dashboards", "export")


def build_fact_daily_summary(txns):
    txns["txn_date"] = pd.to_datetime(txns["timestamp"]).dt.date
    daily = txns.groupby("txn_date").agg(
        txn_count=("transaction_id", "count"),
        total_value=("amount", "sum"),
        fraud_count=("is_fraud", "sum"),
        fraud_value=("amount", lambda s: s[txns.loc[s.index, "is_fraud"]].sum()),
    ).reset_index()
    daily["fraud_rate_pct"] = (daily["fraud_count"] / daily["txn_count"] * 100).round(4)
    daily["rolling_7d_fraud_rate_pct"] = (
        daily["fraud_count"].rolling(7, min_periods=1).sum()
        / daily["txn_count"].rolling(7, min_periods=1).sum() * 100
    ).round(4)
    return daily


def build_dim_customer_risk(txns, customers):
    agg = txns.groupby("customer_id").agg(
        total_txns=("transaction_id", "count"),
        total_value=("amount", "sum"),
        fraud_txns=("is_fraud", "sum"),
        avg_txn_amount=("amount", "mean"),
    ).reset_index()
    agg["fraud_value"] = txns[txns["is_fraud"]].groupby("customer_id")["amount"].sum().reindex(agg["customer_id"]).fillna(0).values
    out = agg.merge(
        customers[["customer_id", "risk_segment", "home_country", "avg_monthly_spend", "is_frequent_traveler", "account_age_days"]],
        on="customer_id", how="left",
    )
    return out


def build_dim_merchant_risk(txns, merchants):
    agg = txns.groupby("merchant_id").agg(
        total_txns=("transaction_id", "count"),
        total_value=("amount", "sum"),
        fraud_txns=("is_fraud", "sum"),
    ).reset_index()
    agg["fraud_value"] = txns[txns["is_fraud"]].groupby("merchant_id")["amount"].sum().reindex(agg["merchant_id"]).fillna(0).values
    agg["fraud_rate_pct"] = (agg["fraud_txns"] / agg["total_txns"] * 100).round(4)
    out = agg.merge(merchants, on="merchant_id", how="left")
    return out


def build_dim_device_risk(txns, devices):
    linked_accounts = devices.groupby("device_id")["customer_id"].nunique().rename("linked_accounts")
    agg = txns.groupby("device_id").agg(
        total_txns=("transaction_id", "count"),
        fraud_txns=("is_fraud", "sum"),
    ).reset_index()
    agg["fraud_value"] = txns[txns["is_fraud"]].groupby("device_id")["amount"].sum().reindex(agg["device_id"]).fillna(0).values
    out = agg.merge(linked_accounts, on="device_id", how="left")
    device_type = devices.drop_duplicates("device_id")[["device_id", "device_type"]]
    out = out.merge(device_type, on="device_id", how="left")
    return out


def build_fact_geographic_summary(txns):
    agg = txns.groupby("txn_country").agg(
        total_txns=("transaction_id", "count"),
        total_value=("amount", "sum"),
        fraud_txns=("is_fraud", "sum"),
    ).reset_index()
    agg["fraud_value"] = txns[txns["is_fraud"]].groupby("txn_country")["amount"].sum().reindex(agg["txn_country"]).fillna(0).values
    agg["fraud_rate_pct"] = (agg["fraud_txns"] / agg["total_txns"] * 100).round(4)
    return agg


def build_fact_investigation_queue(investigations, txns):
    out = investigations.merge(
        txns[["transaction_id", "amount", "is_fraud", "customer_id", "merchant_id"]],
        on="transaction_id", how="left",
    )
    return out


def build_fact_model_performance():
    with open(os.path.join(MODELS_DIR, "model_comparison.json")) as f:
        model_comp = pd.DataFrame(json.load(f))
    return model_comp


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    txns = pd.read_parquet(os.path.join(PROCESSED_DIR, "transactions_features.parquet"))
    customers = pd.read_parquet(os.path.join(RAW_DIR, "customers.parquet"))
    merchants = pd.read_parquet(os.path.join(RAW_DIR, "merchants.parquet"))
    devices = pd.read_parquet(os.path.join(RAW_DIR, "devices.parquet"))
    investigations = pd.read_parquet(os.path.join(RAW_DIR, "investigations.parquet"))

    exports = {
        "fact_daily_summary.csv": build_fact_daily_summary(txns),
        "dim_customer_risk.csv": build_dim_customer_risk(txns, customers),
        "dim_merchant_risk.csv": build_dim_merchant_risk(txns, merchants),
        "dim_device_risk.csv": build_dim_device_risk(txns, devices),
        "fact_geographic_summary.csv": build_fact_geographic_summary(txns),
        "fact_investigation_queue.csv": build_fact_investigation_queue(investigations, txns),
        "fact_model_performance.csv": build_fact_model_performance(),
    }

    # Transaction-level scored fact table (test slice, keeps file size sane)
    scored_path = os.path.join(PROCESSED_DIR, "test_scored_with_anomaly.parquet")
    if os.path.exists(scored_path):
        scored = pd.read_parquet(scored_path)
        keep_cols = [
            "transaction_id", "customer_id", "merchant_id", "timestamp", "amount",
            "txn_country", "is_fraud", "score_logreg", "score_rf", "score_xgb",
            "anomaly_score", "merchant_category", "device_accounts_count",
        ]
        exports["fact_transactions_scored.csv"] = scored[[c for c in keep_cols if c in scored.columns]]

    for filename, df in exports.items():
        path = os.path.join(OUT_DIR, filename)
        df.to_csv(path, index=False)
        print(f"  {filename:35s} {len(df):>8,} rows -> {path}")

    print(f"\nAll Power BI export tables written to {OUT_DIR}")


if __name__ == "__main__":
    main()
