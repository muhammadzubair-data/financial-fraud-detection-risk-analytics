"""
Explainability layer: turns a raw XGBoost fraud probability into something
an investigator (not a data scientist) can act on.

  Fraud probability = 92%          <- what a bare model gives you
      vs.
  Risk Score: 92/100
    - New device
    - Transaction amount 6.4x normal behavior
    - 8 transactions within the last hour
    - Unusual geographic location
    - High-risk merchant

We use SHAP TreeExplainer on the trained XGBoost model for the technical
attribution (which features pushed this specific prediction up/down), then
translate the top contributing features into plain-language, business-
readable reason codes via a lookup table. This keeps the SHAP values as the
ground truth for "why", while the reason codes are just a presentation layer
on top for non-technical stakeholders.
"""

import os
import json
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from config import RAW_DATA_DIR

PROCESSED_DIR = os.path.join(os.path.dirname(RAW_DATA_DIR), "processed")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(RAW_DATA_DIR)), "src", "models", "artifacts")

REASON_CODE_MAP = {
    "amount_vs_customer_avg": "Transaction amount {mult}x this customer's normal behavior",
    "customer_amount_zscore": "Transaction amount far outside this customer's normal range",
    "txns_last_1h": "{val:.0f} transactions within the last hour",
    "txns_last_24h": "{val:.0f} transactions within the last 24 hours",
    "new_device_flag": "New, previously unseen device",
    "device_accounts_count": "Device linked to {val:.0f} different customer accounts",
    "is_shared_device": "Device shared across multiple accounts",
    "distance_from_prev_km": "Transaction location {val:.0f} km from previous transaction",
    "is_cross_border": "Transaction in a country different from customer's home country",
    "failed_attempts_24h": "{val:.0f} failed attempts in the prior 24 hours before this approval",
    "merchant_risk_score": "High-risk merchant category (risk score {val:.1f})",
    "is_high_risk_outlier": "Merchant flagged as a high-risk outlier",
    "is_night_txn": "Transaction occurred during overnight hours",
    "amount": "Unusually large transaction amount (${val:,.0f})",
    "account_age_days": "Relatively new account ({val:.0f} days old)",
}


def reason_from_feature(feat, val, base_val=None):
    # Business reason codes should only be emitted when the feature value
    # actually satisfies the plain-language condition. SHAP may identify
    # account_age_days as influential even for an old account, so do not call
    # it "relatively new" unless the account is <= 90 days old.
    if feat == "account_age_days" and float(val) > 90:
        return None

    template = REASON_CODE_MAP.get(feat)
    if not template:
        return None
    try:
        if "{mult}" in template:
            return template.format(mult=round(val, 1))
        return template.format(val=val)
    except (ValueError, KeyError):
        return template


def main(n_explain: int = 25, top_k_features: int = 5):
    test = pd.read_parquet(os.path.join(PROCESSED_DIR, "test_scored_with_anomaly.parquet"))
    with open(os.path.join(MODELS_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)

    for c in feature_cols:
        if test[c].dtype == bool:
            test[c] = test[c].astype(int)
    X = test[feature_cols].fillna(0)

    model = xgb.XGBClassifier()
    model.load_model(os.path.join(MODELS_DIR, "xgboost.json"))

    print("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)

    # SHAP over the full test set is expensive; explain the highest-risk
    # transactions -- the ones that actually go to an investigator's queue.
    top_risk_idx = test["score_xgb"].values.argsort()[::-1][:max(n_explain, 500)]
    X_top = X.iloc[top_risk_idx]
    shap_values = explainer.shap_values(X_top)

    explanations = []
    for row_pos in range(min(n_explain, len(X_top))):
        row = test.iloc[top_risk_idx[row_pos]]
        sv = shap_values[row_pos]
        feature_contribs = sorted(
            zip(feature_cols, sv, X_top.iloc[row_pos].values),
            key=lambda t: -abs(t[1]),
        )[:top_k_features]

        reasons = []
        for feat, contrib, val in feature_contribs:
            if contrib <= 0:
                continue  # only surface features pushing risk UP
            reason = reason_from_feature(feat, val)
            if reason:
                reasons.append(reason)

        explanations.append({
            "transaction_id": int(row["transaction_id"]),
            "customer_id": int(row["customer_id"]),
            "amount": float(row["amount"]),
            "fraud_probability": round(float(row["score_xgb"]), 4),
            "risk_score_0_100": round(float(row["score_xgb"]) * 100),
            "actually_fraud": bool(row["is_fraud"]),
            "reason_codes": reasons,
        })

    out_path = os.path.join(MODELS_DIR, "sample_explanations.json")
    with open(out_path, "w") as f:
        json.dump(explanations, f, indent=2)

    # Global feature importance from SHAP (mean |SHAP value|) for the
    # investigated slice -- useful for a methodology / model-performance page.
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    global_importance = sorted(
        zip(feature_cols, mean_abs_shap), key=lambda t: -t[1]
    )[:15]
    with open(os.path.join(MODELS_DIR, "global_feature_importance.json"), "w") as f:
        json.dump([{"feature": f, "mean_abs_shap": round(float(v), 5)} for f, v in global_importance], f, indent=2)

    print(f"\nSaved {len(explanations)} sample explanations -> {out_path}")
    print("\nTop 10 global risk drivers (mean |SHAP value|):")
    for feat, val in global_importance[:10]:
        print(f"  {feat:30s} {val:.4f}")

    print("\nExample explanation:")
    print(json.dumps(explanations[0], indent=2))


if __name__ == "__main__":
    main()
