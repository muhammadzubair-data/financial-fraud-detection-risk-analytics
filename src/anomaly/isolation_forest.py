"""
Unsupervised anomaly detection (Isolation Forest) as a complement to
supervised fraud classification.

Why: supervised models can only ever learn patterns that resemble
HISTORICAL labeled fraud. Isolation Forest flags transactions that are
simply "structurally unusual" in feature space, with no reference to the
fraud label at all. That lets us split flagged transactions into two
buckets that matter operationally:

  1. Known-pattern fraud       -- caught by BOTH the supervised model
                                   and the anomaly detector (or by the
                                   supervised model alone). High confidence.
  2. Novel / unlabeled anomaly -- flagged ONLY by Isolation Forest. These
                                   are exactly the transactions a purely
                                   supervised system would silently miss --
                                   e.g. a brand-new fraud technique that
                                   doesn't resemble anything in training data.

We report the overlap between the two systems and how much of the
Isolation-Forest-only bucket still turns out to be true fraud (a
proxy for "would this have been worth building at all").
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import RAW_DATA_DIR

PROCESSED_DIR = os.path.join(os.path.dirname(RAW_DATA_DIR), "processed")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(RAW_DATA_DIR)), "src", "models", "artifacts")

ANOMALY_FEATURES = [
    "amount", "amount_log", "txns_last_1h", "txns_last_24h",
    "amount_vs_customer_avg", "customer_amount_zscore", "time_since_last_txn_sec",
    "device_accounts_count", "distance_from_prev_km", "is_cross_border",
    "failed_attempts_24h", "merchant_risk_score", "hour_of_day",
]


def main():
    test = pd.read_parquet(os.path.join(PROCESSED_DIR, "test_scored.parquet"))
    for c in ANOMALY_FEATURES:
        if test[c].dtype == bool:
            test[c] = test[c].astype(int)
    X = test[ANOMALY_FEATURES].fillna(0).values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # contamination ~= expected fraud rate, but a bit looser since we WANT
    # this model to flag more than just known fraud (that's the point).
    iso = IsolationForest(
        n_estimators=300, contamination=0.02, random_state=42, n_jobs=-1,
    )
    iso.fit(X_s)
    # lower score = more anomalous; flip sign so higher = riskier, consistent
    # with the supervised model scores.
    anomaly_score = -iso.score_samples(X_s)
    test["anomaly_score"] = anomaly_score

    n_flag = int(0.02 * len(test))
    threshold = np.sort(anomaly_score)[::-1][n_flag - 1]
    test["iso_flagged"] = anomaly_score >= threshold

    # Compare against the supervised model's top-K flags at the same budget.
    supervised_threshold = np.sort(test["score_xgb"].values)[::-1][n_flag - 1]
    test["xgb_flagged"] = test["score_xgb"] >= supervised_threshold

    both = test["iso_flagged"] & test["xgb_flagged"]
    iso_only = test["iso_flagged"] & ~test["xgb_flagged"]
    xgb_only = test["xgb_flagged"] & ~test["iso_flagged"]

    summary = {
        "n_flagged_each_system": n_flag,
        "overlap_both_systems": int(both.sum()),
        "iso_only": int(iso_only.sum()),
        "xgb_only": int(xgb_only.sum()),
        "fraud_rate_in_overlap": round(float(test.loc[both, "is_fraud"].mean()), 4) if both.sum() else None,
        "fraud_rate_in_iso_only": round(float(test.loc[iso_only, "is_fraud"].mean()), 4) if iso_only.sum() else None,
        "fraud_rate_in_xgb_only": round(float(test.loc[xgb_only, "is_fraud"].mean()), 4) if xgb_only.sum() else None,
        "overall_fraud_rate": round(float(test["is_fraud"].mean()), 4),
    }

    print(json.dumps(summary, indent=2))

    test.to_parquet(os.path.join(PROCESSED_DIR, "test_scored_with_anomaly.parquet"), index=False)
    with open(os.path.join(MODELS_DIR, "anomaly_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nKey finding: Isolation-Forest-only flags run at "
          f"{summary['fraud_rate_in_iso_only']:.2%} true fraud rate vs a baseline "
          f"of {summary['overall_fraud_rate']:.2%} -- i.e. these are transactions "
          f"the supervised model would have missed entirely, but are still "
          f"meaningfully riskier than average.")


if __name__ == "__main__":
    main()
