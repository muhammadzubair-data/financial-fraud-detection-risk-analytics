"""
Point-in-time, leakage-safe feature engineering.

RULE: every engineered feature for transaction T may only use information
that would have been available strictly BEFORE T occurred:
  - prior transactions of the same customer (expanding/rolling stats,
    computed on a chronologically-sorted, per-customer basis, always
    excluding the current row itself)
  - static entity attributes fixed at generation time (merchant risk
    score, device metadata, customer signup info) -- these are legitimate
    "known in advance" attributes, not future information

The raw `_reason_codes` and `is_fraud` columns are the label / label-adjacent
columns and are kept separate from the feature matrix (is_fraud is the target,
_reason_codes is dropped entirely before modeling -- it's a data-generation
validation artifact, not something a real fraud team would have).
"""

import numpy as np
import pandas as pd

from config import RAW_DATA_DIR
import os

PROCESSED_DIR = os.path.join(os.path.dirname(RAW_DATA_DIR), "processed")


def _velocity_counts(times_sec: np.ndarray, window_sec: int) -> np.ndarray:
    """
    For each index j in a chronologically sorted array of per-customer
    timestamps (seconds), count how many STRICTLY PRIOR transactions
    (indices < j) fall within `window_sec` before times_sec[j].
    """
    n = len(times_sec)
    out = np.zeros(n, dtype=np.int32)
    for j in range(n):
        t = times_sec[j]
        lo = np.searchsorted(times_sec[:j], t - window_sec, side="left")
        out[j] = j - lo
    return out


def add_customer_level_features(txns: pd.DataFrame) -> pd.DataFrame:
    txns = txns.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

    n = len(txns)
    txns_last_1h = np.zeros(n, dtype=np.int32)
    txns_last_24h = np.zeros(n, dtype=np.int32)
    amount_vs_customer_avg = np.zeros(n, dtype=np.float64)
    customer_amount_zscore = np.zeros(n, dtype=np.float64)
    time_since_last_txn_sec = np.full(n, -1.0, dtype=np.float64)

    ts_sec_all = txns["timestamp"].values.astype("datetime64[s]").astype(np.int64)
    amounts_all = txns["amount"].values.astype(np.float64)

    for cust_id, idx in txns.groupby("customer_id", sort=False).indices.items():
        idx = np.asarray(idx)
        # groupby indices are already in original (sorted) row order for this customer
        t_sec = ts_sec_all[idx]
        amt = amounts_all[idx]

        v1h = _velocity_counts(t_sec, 3600)
        v24h = _velocity_counts(t_sec, 86400)
        txns_last_1h[idx] = v1h
        txns_last_24h[idx] = v24h

        # expanding mean/std of PRIOR transactions only (shift by 1)
        cs = np.cumsum(amt)
        csq = np.cumsum(amt ** 2)
        for j in range(len(idx)):
            if j == 0:
                prior_mean = amt[0] / 20.0  # cold-start prior, mirrors generator's bootstrap
                prior_std = max(prior_mean * 0.5, 1.0)
                time_since_last_txn_sec[idx[j]] = -1.0
            else:
                prior_n = j
                prior_sum = cs[j - 1]
                prior_sumsq = csq[j - 1]
                prior_mean = prior_sum / prior_n
                prior_var = max(prior_sumsq / prior_n - prior_mean ** 2, 0.0)
                prior_std = max(np.sqrt(prior_var), 1.0)
                time_since_last_txn_sec[idx[j]] = float(t_sec[j] - t_sec[j - 1])

            amount_vs_customer_avg[idx[j]] = amt[j] / max(prior_mean, 1e-6)
            customer_amount_zscore[idx[j]] = (amt[j] - prior_mean) / prior_std

    txns["txns_last_1h"] = txns_last_1h
    txns["txns_last_24h"] = txns_last_24h
    txns["amount_vs_customer_avg"] = np.round(amount_vs_customer_avg, 4)
    txns["customer_amount_zscore"] = np.round(customer_amount_zscore, 4)
    txns["time_since_last_txn_sec"] = time_since_last_txn_sec
    txns["is_first_txn"] = (time_since_last_txn_sec < 0).astype(int)
    return txns


def add_static_entity_features(txns: pd.DataFrame, customers, merchants, devices) -> pd.DataFrame:
    device_acct_counts = devices.groupby("device_id")["customer_id"].nunique().rename("device_accounts_count")
    txns = txns.merge(device_acct_counts, on="device_id", how="left")
    txns["device_accounts_count"] = txns["device_accounts_count"].fillna(1).astype(int)
    txns["is_shared_device"] = (txns["device_accounts_count"] > 1).astype(int)

    cust_cols = customers[[
        "customer_id", "account_age_days", "home_country", "avg_monthly_spend",
        "is_frequent_traveler",
    ]]
    txns = txns.merge(cust_cols, on="customer_id", how="left")

    merch_cols = merchants[["merchant_id", "merchant_category", "merchant_risk_score", "is_high_risk_outlier"]]
    txns = txns.merge(merch_cols, on="merchant_id", how="left")

    txns["is_cross_border"] = (txns["txn_country"] != txns["home_country"]).astype(int)

    ts = pd.to_datetime(txns["timestamp"])
    txns["hour_of_day"] = ts.dt.hour
    txns["day_of_week"] = ts.dt.dayofweek
    txns["is_night_txn"] = txns["hour_of_day"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    txns["amount_log"] = np.log1p(txns["amount"])

    return txns


def build(save: bool = True) -> pd.DataFrame:
    print("Loading raw tables...")
    txns = pd.read_parquet(os.path.join(RAW_DATA_DIR, "transactions.parquet"))
    customers = pd.read_parquet(os.path.join(RAW_DATA_DIR, "customers.parquet"))
    merchants = pd.read_parquet(os.path.join(RAW_DATA_DIR, "merchants.parquet"))
    devices = pd.read_parquet(os.path.join(RAW_DATA_DIR, "devices.parquet"))

    print(f"Building velocity + expanding-stat features for {len(txns):,} transactions "
          f"across {txns['customer_id'].nunique():,} customers...")
    txns = add_customer_level_features(txns)

    print("Joining static entity features (device, customer, merchant)...")
    txns = add_static_entity_features(txns, customers, merchants, devices)

    # Drop the data-generation-only validation column; keep is_fraud as the target.
    feature_df = txns.drop(columns=["_reason_codes"], errors="ignore")
    feature_df = feature_df.sort_values("transaction_id").reset_index(drop=True)

    if save:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        out_path = os.path.join(PROCESSED_DIR, "transactions_features.parquet")
        feature_df.to_parquet(out_path, index=False)
        print(f"Saved {len(feature_df):,} rows x {feature_df.shape[1]} cols -> {out_path}")

    return feature_df


if __name__ == "__main__":
    df = build()
    print("\nFraud rate:", df["is_fraud"].mean())
    print("\nSample engineered feature columns:")
    new_cols = [
        "txns_last_1h", "txns_last_24h", "amount_vs_customer_avg", "customer_amount_zscore",
        "time_since_last_txn_sec", "device_accounts_count", "is_shared_device",
        "is_cross_border", "merchant_risk_score", "hour_of_day", "amount_log",
    ]
    print(df[new_cols].describe().T)
