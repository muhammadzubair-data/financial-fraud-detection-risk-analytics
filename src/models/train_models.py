"""
Supervised fraud classification: Logistic Regression -> Random Forest -> XGBoost.

Deliberately does NOT lead with accuracy (useless at a 0.6% fraud rate --
predicting "never fraud" scores 99.4%). Instead we evaluate:

  PR-AUC          area under precision-recall curve (the right headline
                  metric for severe class imbalance)
  ROC-AUC         included for completeness / comparability
  Recall@K        of the top-K riskiest transactions (K = a fixed review
                  capacity for the evaluation slice), what fraction of all fraud do we
                  actually catch?
  Precision@K     of those top-K flagged transactions, what fraction are
                  really fraud? (investigator time wasted on false alarms)
  Dollar-weighted capture  of the top-K by risk, what fraction of total
                  fraud DOLLAR VALUE would be caught -- ties directly into
                  the investigation-optimization layer built next.

Split is chronological (train on earlier transactions, test on the most
recent slice) rather than random, since random splits leak future
customer behavior into training in a time-series fraud setting.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
)
import xgboost as xgb

from config import RAW_DATA_DIR

PROCESSED_DIR = os.path.join(os.path.dirname(RAW_DATA_DIR), "processed")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(RAW_DATA_DIR)), "src", "models", "artifacts")

FEATURE_COLS = [
    "amount", "amount_log", "txns_last_1h", "txns_last_24h",
    "amount_vs_customer_avg", "customer_amount_zscore", "time_since_last_txn_sec",
    "is_first_txn", "device_accounts_count", "is_shared_device", "new_device_flag",
    "distance_from_prev_km", "is_cross_border", "failed_attempts_24h",
    "merchant_risk_score", "is_high_risk_outlier", "account_age_days",
    "avg_monthly_spend", "is_frequent_traveler", "hour_of_day", "day_of_week",
    "is_night_txn",
]
CATEGORICAL_COLS = ["merchant_category"]
TARGET = "is_fraud"


def load_split():
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "transactions_features.parquet"))
    df = df.sort_values("timestamp").reset_index(drop=True)

    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, prefix="cat")
    cat_dummy_cols = [c for c in df.columns if c.startswith("cat_")]
    feature_cols = FEATURE_COLS + cat_dummy_cols

    for c in feature_cols:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
    df[feature_cols] = df[feature_cols].fillna(0)

    # Chronological 70/15/15 split: train / validation / test.
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    print(f"Train: {len(train):,} ({train['timestamp'].min()} - {train['timestamp'].max()}) "
          f"fraud rate {train[TARGET].mean():.4%}")
    print(f"Val:   {len(val):,} ({val['timestamp'].min()} - {val['timestamp'].max()}) "
          f"fraud rate {val[TARGET].mean():.4%}")
    print(f"Test:  {len(test):,} ({test['timestamp'].min()} - {test['timestamp'].max()}) "
          f"fraud rate {test[TARGET].mean():.4%}")

    return train, val, test, feature_cols


def recall_precision_at_k(y_true, y_scores, amounts, k):
    order = np.argsort(-y_scores)
    top_k_idx = order[:k]
    y_true = np.asarray(y_true)
    n_fraud_total = y_true.sum()
    n_fraud_in_topk = y_true[top_k_idx].sum()
    recall_at_k = n_fraud_in_topk / max(n_fraud_total, 1)
    precision_at_k = n_fraud_in_topk / max(k, 1)

    total_fraud_dollars = amounts[y_true.astype(bool)].sum()
    captured_fraud_dollars = amounts[top_k_idx][y_true[top_k_idx].astype(bool)].sum()
    dollar_capture_at_k = captured_fraud_dollars / max(total_fraud_dollars, 1)

    return recall_at_k, precision_at_k, dollar_capture_at_k


def evaluate(name, y_true, y_scores, amounts, k):
    pr_auc = average_precision_score(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    recall_k, precision_k, dollar_capture_k = recall_precision_at_k(y_true, y_scores, amounts, k)
    result = {
        "model": name,
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        f"recall_at_{k}": round(recall_k, 4),
        f"precision_at_{k}": round(precision_k, 4),
        f"dollar_capture_at_{k}": round(dollar_capture_k, 4),
    }
    print(f"  {name:20s} PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}  "
          f"Recall@{k}={recall_k:.4f}  Precision@{k}={precision_k:.4f}  "
          f"$Capture@{k}={dollar_capture_k:.4f}")
    return result


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    train, val, test, feature_cols = load_split()

    X_train, y_train = train[feature_cols].values, train[TARGET].values
    X_val, y_val = val[feature_cols].values, val[TARGET].values
    X_test, y_test = test[feature_cols].values, test[TARGET].values
    amounts_test = test["amount"].values

    # Fixed review capacity used for Recall@K / Precision@K comparability
    # across models -- roughly 2% of the test slice, a deliberately tight
    # evaluation-period operations constraint.
    K = max(int(0.02 * len(test)), 50)
    print(f"\nUsing fixed review capacity K={K} for the test slice.\n")

    results = []

    # --- 1. Logistic Regression (scaled, class_weight balanced) ---
    print("Training Logistic Regression...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
    lr.fit(X_train_s, y_train)
    lr_scores = lr.predict_proba(X_test_s)[:, 1]
    results.append(evaluate("LogisticRegression", y_test, lr_scores, amounts_test, K))

    # --- 2. Random Forest (balanced subsample) ---
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_scores = rf.predict_proba(X_test)[:, 1]
    results.append(evaluate("RandomForest", y_test, rf_scores, amounts_test, K))

    # --- 3. XGBoost (scale_pos_weight for imbalance) ---
    print("Training XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        random_state=42, n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_scores = xgb_model.predict_proba(X_test)[:, 1]
    results.append(evaluate("XGBoost", y_test, xgb_scores, amounts_test, K))

    # Persist everything the next stages (anomaly, optimization, explainability) need.
    import joblib
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(lr, os.path.join(MODELS_DIR, "logreg.joblib"))
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.joblib"))
    xgb_model.save_model(os.path.join(MODELS_DIR, "xgboost.json"))
    with open(os.path.join(MODELS_DIR, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)

    test_out = test.copy()
    test_out["score_logreg"] = lr_scores
    test_out["score_rf"] = rf_scores
    test_out["score_xgb"] = xgb_scores
    test_out.to_parquet(os.path.join(PROCESSED_DIR, "test_scored.parquet"), index=False)

    with open(os.path.join(MODELS_DIR, "model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved model artifacts to {MODELS_DIR}")
    print(f"Saved scored test set ({len(test_out):,} rows) -> {PROCESSED_DIR}/test_scored.parquet")

    best = max(results, key=lambda r: r["pr_auc"])
    print(f"\nBest model by PR-AUC: {best['model']} ({best['pr_auc']})")


if __name__ == "__main__":
    main()
