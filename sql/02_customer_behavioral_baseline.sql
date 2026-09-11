-- Customer behavioral baselines & deviations
-- Answers: which customers are transacting well outside their own normal pattern?

-- 2a. Per-customer baseline stats vs. realized fraud exposure
SELECT
    c.customer_id,
    c.risk_segment,
    c.avg_monthly_spend,
    COUNT(t.transaction_id)                                  AS total_txns,
    ROUND(AVG(t.amount), 2)                                  AS avg_txn_amount,
    ROUND(MAX(t.amount), 2)                                  AS max_txn_amount,
    SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END)               AS fraud_txns,
    ROUND(SUM(CASE WHEN t.is_fraud THEN t.amount ELSE 0 END), 2) AS fraud_value
FROM customers c
JOIN transactions t ON t.customer_id = c.customer_id
GROUP BY c.customer_id, c.risk_segment, c.avg_monthly_spend
HAVING SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) > 0
ORDER BY fraud_value DESC
LIMIT 100;

-- 2b. Fraud rate by latent risk segment (validates the segment actually
-- carries signal -- useful sanity check / methodology appendix)
SELECT
    c.risk_segment,
    COUNT(t.transaction_id)                     AS txn_count,
    SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END)  AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) / COUNT(t.transaction_id), 4) AS fraud_rate_pct
FROM customers c
JOIN transactions t ON t.customer_id = c.customer_id
GROUP BY c.risk_segment
ORDER BY fraud_rate_pct DESC;

-- 2c. Customers whose transactions deviate most from their own average
-- (large customer_amount_zscore = highly unusual FOR THEM specifically,
-- as opposed to unusual in absolute dollar terms)
SELECT
    customer_id,
    transaction_id,
    timestamp,
    amount,
    ROUND(customer_amount_zscore, 2) AS amount_zscore_vs_own_history,
    is_fraud
FROM transactions
WHERE customer_amount_zscore > 5
ORDER BY customer_amount_zscore DESC
LIMIT 100;

-- 2d. New-account fraud exposure (accounts under 90 days old)
SELECT
    CASE
        WHEN account_age_days < 30 THEN '0-29 days'
        WHEN account_age_days < 90 THEN '30-89 days'
        WHEN account_age_days < 365 THEN '90-364 days'
        ELSE '365+ days'
    END AS account_age_bucket,
    COUNT(*) AS txn_count,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 4) AS fraud_rate_pct
FROM transactions
GROUP BY account_age_bucket
ORDER BY fraud_rate_pct DESC;
