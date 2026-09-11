"""
Generates the two "outcome" tables that sit downstream of transactions:

  chargebacks     -- disputes filed after the fact. Not every fraud becomes
                     a chargeback (detection lag / undetected fraud), and
                     not every chargeback is fraud (legitimate disputes:
                     product not received, duplicate charge, etc.). This
                     mismatch is what makes "chargeback losses" a distinct,
                     imperfect proxy for "true fraud" -- exactly like real
                     fintech data.

  investigations  -- represents a HISTORICAL fraud-ops review queue, built
                     from simple rule-based heuristics (the kind of naive
                     rules a team might use *before* a model exists). This
                     gives us realistic base rates for analyst workload and
                     a "false positive / false negative" story even before
                     any ML model is introduced. The ML-driven investigation
                     ranking (Expected Fraud Loss) is a separate, later
                     layer that competes against this historical baseline.
"""

import numpy as np
import pandas as pd
from datetime import timedelta

from config import SEED, RAW_DATA_DIR

rng = np.random.default_rng(SEED + 2)

CHARGEBACK_REASONS_NON_FRAUD = ["product_not_received", "duplicate_charge", "subscription_dispute", "other"]


def generate_chargebacks(txns: pd.DataFrame) -> pd.DataFrame:
    fraud_txns = txns[txns["is_fraud"]]
    legit_txns = txns[~txns["is_fraud"]]

    # Only a fraction of true fraud ever surfaces as a chargeback within the
    # data window (some is caught by other means, some never disputed).
    fraud_cb = fraud_txns.sample(frac=0.72, random_state=SEED)

    # A small fraction of legitimate transactions also get disputed --
    # normal e-commerce friction, unrelated to fraud.
    legit_cb = legit_txns.sample(frac=0.004, random_state=SEED)

    rows = []
    cb_id = 1
    for _, txn in fraud_cb.iterrows():
        lag_days = rng.integers(3, 75)
        rows.append({
            "chargeback_id": cb_id,
            "transaction_id": txn["transaction_id"],
            "chargeback_date": txn["timestamp"] + timedelta(days=int(lag_days)),
            "chargeback_reason": "fraud",
            "chargeback_amount": txn["amount"],
            "is_fraud_related": True,
        })
        cb_id += 1

    for _, txn in legit_cb.iterrows():
        lag_days = rng.integers(1, 45)
        reason = rng.choice(CHARGEBACK_REASONS_NON_FRAUD)
        rows.append({
            "chargeback_id": cb_id,
            "transaction_id": txn["transaction_id"],
            "chargeback_date": txn["timestamp"] + timedelta(days=int(lag_days)),
            "chargeback_reason": reason,
            "chargeback_amount": txn["amount"],
            "is_fraud_related": False,
        })
        cb_id += 1

    return pd.DataFrame(rows).sort_values("chargeback_date").reset_index(drop=True)


def generate_investigations(txns: pd.DataFrame) -> pd.DataFrame:
    """
    Simple, deliberately-naive rule-based flagging to represent what a
    fraud ops team might have used historically (pre-ML): flag on amount
    thresholds, new device, or high merchant risk. This produces a queue
    with realistic precision (lots of false positives) that the later
    ML-based investigation-priority model will be benchmarked against.
    """
    df = txns.copy()

    flag_score = (
        (df["amount"] > df["amount"].quantile(0.97)).astype(int) * 1
        + df["new_device_flag"].astype(int) * 1
        + (df["failed_attempts_24h"] > 0).astype(int) * 1
        + (df["distance_from_prev_km"] > 3000).astype(int) * 1
    )
    # add a little randomness so the naive rule isn't perfectly deterministic
    flagged_mask = (flag_score >= 2) | (rng.random(len(df)) < 0.001)
    flagged = df[flagged_mask].copy()

    # Investigation capacity constraint: ops can only review so many per day.
    # Keep at most ~60 per day, prioritized crudely by flag_score (mimics a
    # simple, non-optimized historical prioritization -- this is exactly
    # the naive baseline the Expected-Fraud-Loss ranking will later beat).
    flagged["flag_score"] = flag_score[flagged_mask]
    flagged["review_date"] = flagged["timestamp"].dt.date

    kept_rows = []
    for date, group in flagged.groupby("review_date"):
        capped = group.sort_values("flag_score", ascending=False).head(60)
        kept_rows.append(capped)
    flagged = pd.concat(kept_rows) if kept_rows else flagged.iloc[0:0]

    rows = []
    inv_id = 1
    for _, txn in flagged.iterrows():
        is_actually_fraud = txn["is_fraud"]
        # Analysts aren't perfect: some true fraud gets missed (false
        # negative at review time), most legitimate flags get cleared.
        if is_actually_fraud:
            outcome = "confirmed_fraud" if rng.random() < 0.82 else "false_negative_cleared"
        else:
            outcome = "false_positive_cleared" if rng.random() < 0.95 else "confirmed_fraud"  # rare analyst error

        resolution_hours = rng.gamma(shape=2.0, scale=6.0)  # right-skewed review time
        rows.append({
            "investigation_id": inv_id,
            "transaction_id": txn["transaction_id"],
            "flagged_date": txn["timestamp"],
            "flag_score": int(txn["flag_score"]),
            "outcome": outcome,
            "resolution_time_hours": round(float(resolution_hours), 2),
        })
        inv_id += 1

    return pd.DataFrame(rows).sort_values("flagged_date").reset_index(drop=True)


if __name__ == "__main__":
    print("Loading transactions...")
    txns = pd.read_parquet(RAW_DATA_DIR + "/transactions.parquet")

    print("Generating chargebacks...")
    chargebacks = generate_chargebacks(txns)
    chargebacks.to_parquet(RAW_DATA_DIR + "/chargebacks.parquet", index=False)

    print("Generating investigations (historical rule-based queue)...")
    investigations = generate_investigations(txns)
    investigations.to_parquet(RAW_DATA_DIR + "/investigations.parquet", index=False)

    print(f"\nchargebacks: {len(chargebacks):,} "
          f"({chargebacks['is_fraud_related'].mean():.1%} fraud-related)")
    print(f"investigations: {len(investigations):,}")
    print(investigations["outcome"].value_counts(normalize=True).round(3))

    true_fraud = txns["is_fraud"].sum()
    caught = investigations[investigations["outcome"] == "confirmed_fraud"]["transaction_id"].nunique()
    print(f"\nTrue fraud transactions: {true_fraud:,}")
    print(f"Caught by historical rule-based queue: {caught:,} "
          f"({caught/true_fraud:.1%} of all fraud)")
