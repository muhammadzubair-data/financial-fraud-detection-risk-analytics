"""
Investigation prioritization and leakage-safe operating-threshold evaluation.

Flagship business question:
  With a fixed review budget for an evaluation period, which cases maximize
  prevented financial loss?

Expected Fraud Loss = P(fraud) * transaction_value * expected_loss_rate

The ranking comparison is evaluated on the final policy holdout. The operating
threshold is selected on an earlier policy-tuning slice and then frozen before
being evaluated on the later holdout. This avoids selecting and advertising a
threshold on the same observations used for final policy performance.
"""

import os
import json
import numpy as np
import pandas as pd

from config import RAW_DATA_DIR

PROCESSED_DIR = os.path.join(os.path.dirname(RAW_DATA_DIR), "processed")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(RAW_DATA_DIR)), "src", "models", "artifacts")

EXPECTED_LOSS_RATE = 0.85
COST_PER_INVESTIGATION = 12.0
FRICTION_COST_PER_FALSE_POSITIVE = 8.0

# Budgets apply to the evaluation period, not per day.
INVESTIGATION_BUDGETS = [250, 500, 1000, 1500, 2500]


def expected_fraud_loss_ranking(df, score_col, budget):
    d = df.copy()
    d["expected_fraud_loss"] = d[score_col] * d["amount"] * EXPECTED_LOSS_RATE
    return d.sort_values("expected_fraud_loss", ascending=False).head(budget)


def probability_only_ranking(df, score_col, budget):
    return df.sort_values(score_col, ascending=False).head(budget)


def money_protected(selected):
    fraud_rows = selected[selected["is_fraud"].astype(bool)]
    return float(fraud_rows["amount"].sum() * EXPECTED_LOSS_RATE)


def compare_ranking_strategies(df, score_col="score_xgb"):
    results = []
    for budget in INVESTIGATION_BUDGETS:
        if budget > len(df):
            continue
        eloss_selected = expected_fraud_loss_ranking(df, score_col, budget)
        prob_selected = probability_only_ranking(df, score_col, budget)
        eloss_money = money_protected(eloss_selected)
        prob_money = money_protected(prob_selected)
        lift_pct = (eloss_money - prob_money) / max(prob_money, 1.0) * 100
        results.append({
            "investigation_budget": budget,
            "money_protected_expected_loss_ranking": round(eloss_money, 2),
            "money_protected_probability_only_ranking": round(prob_money, 2),
            "dollar_lift_pct_from_expected_loss_ranking": round(lift_pct, 1),
            "fraud_cases_caught_expected_loss_ranking": int(eloss_selected["is_fraud"].sum()),
            "fraud_cases_caught_probability_only_ranking": int(prob_selected["is_fraud"].sum()),
        })
    return results


def threshold_business_value(df, threshold, score_col="score_xgb"):
    flagged = df[df[score_col] >= threshold]
    true_positives = flagged[flagged["is_fraud"].astype(bool)]
    false_positives = flagged[~flagged["is_fraud"].astype(bool)]
    fraud_prevented = float(true_positives["amount"].sum() * EXPECTED_LOSS_RATE)
    investigation_cost = len(flagged) * COST_PER_INVESTIGATION
    friction_cost = len(false_positives) * FRICTION_COST_PER_FALSE_POSITIVE
    return {
        "threshold": round(float(threshold), 3),
        "n_flagged": int(len(flagged)),
        "fraud_prevented": round(fraud_prevented, 2),
        "investigation_cost": round(investigation_cost, 2),
        "friction_cost": round(friction_cost, 2),
        "net_business_value": round(fraud_prevented - investigation_cost - friction_cost, 2),
        "precision": round(len(true_positives) / max(len(flagged), 1), 4),
        "recall": round(len(true_positives) / max(int(df["is_fraud"].sum()), 1), 4),
    }


def select_threshold(policy_tuning, score_col="score_xgb"):
    rows = [threshold_business_value(policy_tuning, t, score_col)
            for t in np.linspace(0.05, 0.95, 19)]
    result_df = pd.DataFrame(rows)
    best_row = result_df.loc[result_df["net_business_value"].idxmax()].to_dict()
    return result_df, best_row


def main():
    # This file is also exported as dashboards/export/fact_transactions_scored.csv
    # for lightweight GitHub reproduction. The parquet remains the canonical
    # pipeline artifact when available.
    parquet_path = os.path.join(PROCESSED_DIR, "test_scored_with_anomaly.parquet")
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    else:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(PROCESSED_DIR)),
                                "dashboards", "export", "fact_transactions_scored.csv")
        df = pd.read_csv(csv_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Split the original chronological model-test slice again for policy
    # development: earlier half tunes the threshold; later half is untouched
    # until final policy evaluation.
    split = len(df) // 2
    policy_tuning = df.iloc[:split].copy()
    policy_holdout = df.iloc[split:].copy()

    threshold_grid, selected = select_threshold(policy_tuning, "score_xgb")
    frozen_threshold = float(selected["threshold"])
    holdout_frozen = threshold_business_value(policy_holdout, frozen_threshold, "score_xgb")
    holdout_default = threshold_business_value(policy_holdout, 0.5, "score_xgb")
    ranking_results = compare_ranking_strategies(policy_holdout, "score_xgb")

    out = {
        "methodology": {
            "policy_tuning_rows": int(len(policy_tuning)),
            "policy_holdout_rows": int(len(policy_holdout)),
            "policy_tuning_start": str(policy_tuning["timestamp"].min()),
            "policy_tuning_end": str(policy_tuning["timestamp"].max()),
            "policy_holdout_start": str(policy_holdout["timestamp"].min()),
            "policy_holdout_end": str(policy_holdout["timestamp"].max()),
            "note": "Threshold selected on earlier policy-tuning half and frozen before later policy-holdout evaluation."
        },
        "selected_threshold_on_policy_tuning": selected,
        "frozen_threshold_performance_on_policy_holdout": holdout_frozen,
        "default_0_5_performance_on_policy_holdout": holdout_default,
        "ranking_strategy_comparison_on_policy_holdout": ranking_results,
        "assumptions": {
            "expected_loss_rate": EXPECTED_LOSS_RATE,
            "cost_per_investigation": COST_PER_INVESTIGATION,
            "friction_cost_per_false_positive": FRICTION_COST_PER_FALSE_POSITIVE,
        },
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "policy_evaluation.json"), "w") as f:
        json.dump(out, f, indent=2)
    threshold_grid.to_csv(os.path.join(PROCESSED_DIR, "threshold_policy_tuning.csv"), index=False)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
