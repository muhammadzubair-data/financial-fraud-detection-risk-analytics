# Fraud Risk Command Center — DAX Measures & Page Layout

Data source: CSVs in `dashboards/export/` (or the equivalent parquet files
in `data/processed/`). Load each as its own table; relationships below.

## Model / relationships

- `fact_daily_summary[txn_date]` — standalone calendar fact, no relationship needed for page 1-2 visuals.
- `fact_transactions_scored[customer_id]` → `dim_customer_risk[customer_id]` (many-to-one)
- `fact_transactions_scored[merchant_id]` → `dim_merchant_risk[merchant_id]` (many-to-one)
- `fact_investigation_queue[transaction_id]` → `fact_transactions_scored[transaction_id]` (one-to-one, optional)

## Core DAX measures

```dax
Total Transactions = COUNTROWS(fact_transactions_scored)

Total Transaction Value = SUM(fact_transactions_scored[amount])

Fraud Transactions = CALCULATE([Total Transactions], fact_transactions_scored[is_fraud] = TRUE)

Fraud Value = CALCULATE([Total Transaction Value], fact_transactions_scored[is_fraud] = TRUE)

Fraud Rate % = DIVIDE([Fraud Transactions], [Total Transactions], 0) * 100

-- Money protected under a fixed investigation budget (the flagship metric --
-- see src/optimization/investigation_optimizer.py for the underlying calc).
-- Pull this as a static value per budget scenario from
-- src/models/artifacts/policy_evaluation.json, or recompute live:
Expected Fraud Loss =
    fact_transactions_scored[score_xgb] * fact_transactions_scored[amount] * 0.85

Money Protected (Top K) =
VAR K = 2000
VAR RankedTable =
    TOPN(K, fact_transactions_scored, [Expected Fraud Loss], DESC)
RETURN
    SUMX(FILTER(RankedTable, fact_transactions_scored[is_fraud] = TRUE), fact_transactions_scored[amount]) * 0.85

Investigation Precision % =
    DIVIDE(
        CALCULATE(COUNTROWS(fact_investigation_queue), fact_investigation_queue[is_fraud] = TRUE),
        COUNTROWS(fact_investigation_queue), 0
    ) * 100

Rolling 7D Fraud Rate % = AVERAGE(fact_daily_summary[rolling_7d_fraud_rate_pct])

Chargeback Loss Ratio = DIVIDE([Fraud Value Charged Back], [Fraud Value], 0)
```

## Page layout

1. **Executive Overview** — KPI cards (Total Transactions, Fraud Rate %,
   Fraud Value, Money Protected @ current budget), trend sparkline from
   `fact_daily_summary`.
2. **Fraud Trends** — daily/rolling-7D fraud rate line chart, fraud value
   by day, day-of-week / hour-of-day heatmap (from `fact_transactions_scored`).
3. **Transaction Risk** — score distribution histogram (`score_xgb`),
   scatter of amount vs. fraud probability, top-N highest-risk open
   transactions table.
4. **Customer Risk** — `dim_customer_risk` table with fraud value/txns,
   risk segment breakdown, top-N riskiest customers.
5. **Merchant Analysis** — `dim_merchant_risk` fraud rate by category,
   top risky merchants, high-risk-outlier flag distribution.
6. **Device & Geographic Risk** — `dim_device_risk` shared-device table,
   `fact_geographic_summary` map/bar by country.
7. **Investigation Queue** — `fact_investigation_queue` outcome funnel
   (flagged → confirmed fraud / false positive), resolution-time
   distribution, historical rule-based precision vs. the ML-ranked
   approach (static comparison card from `investigation_optimization.json`).
8. **Model Performance** — `fact_model_performance` bar chart (PR-AUC,
   ROC-AUC, Recall@K, Precision@K, $Capture@K across LR / RF / XGBoost),
   plus the threshold-vs-net-business-value line chart from
   `data/processed/threshold_policy_tuning.csv`.
