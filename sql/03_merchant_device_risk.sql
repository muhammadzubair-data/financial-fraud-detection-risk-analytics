-- Merchant & device risk analysis
-- Answers: where is fraud concentrated by merchant category and device sharing?

-- 3a. Fraud rate and exposure by merchant category
SELECT
    m.merchant_category,
    COUNT(t.transaction_id)                                    AS txn_count,
    ROUND(SUM(t.amount), 2)                                     AS total_value,
    SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END)                 AS fraud_count,
    ROUND(SUM(CASE WHEN t.is_fraud THEN t.amount ELSE 0 END), 2) AS fraud_value,
    ROUND(100.0 * SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) / COUNT(t.transaction_id), 4) AS fraud_rate_pct
FROM merchants m
JOIN transactions t ON t.merchant_id = m.merchant_id
GROUP BY m.merchant_category
ORDER BY fraud_rate_pct DESC;

-- 3b. Top 25 riskiest individual merchants by realized fraud value
SELECT
    m.merchant_id,
    m.merchant_category,
    m.merchant_risk_score,
    m.is_high_risk_outlier,
    COUNT(t.transaction_id) AS txn_count,
    SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(SUM(CASE WHEN t.is_fraud THEN t.amount ELSE 0 END), 2) AS fraud_value
FROM merchants m
JOIN transactions t ON t.merchant_id = m.merchant_id
GROUP BY m.merchant_id, m.merchant_category, m.merchant_risk_score, m.is_high_risk_outlier
HAVING fraud_count > 0
ORDER BY fraud_value DESC
LIMIT 25;

-- 3c. Device sharing risk: devices linked to multiple customer accounts
SELECT
    d.device_id,
    d.device_type,
    COUNT(DISTINCT d.customer_id) AS linked_accounts,
    COUNT(t.transaction_id)       AS txn_count,
    SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(SUM(CASE WHEN t.is_fraud THEN t.amount ELSE 0 END), 2) AS fraud_value
FROM devices d
JOIN transactions t ON t.device_id = d.device_id
GROUP BY d.device_id, d.device_type
HAVING COUNT(DISTINCT d.customer_id) > 1
ORDER BY linked_accounts DESC, fraud_value DESC
LIMIT 50;

-- 3d. Shared-device fraud rate vs. single-owner device fraud rate
-- (justifies device_accounts_count / is_shared_device as a model feature)
SELECT
    CASE WHEN is_shared_device = 1 THEN 'shared_device' ELSE 'single_owner_device' END AS device_group,
    COUNT(*) AS txn_count,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 4) AS fraud_rate_pct
FROM transactions
GROUP BY device_group;
