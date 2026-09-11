-- Geographic risk & investigation queue analysis

-- 5a. Fraud rate by transaction country (surfaces genuine hotspots, but see
-- 5b for the traveler-vs-fraud distinction -- geography alone overstates risk)
SELECT
    txn_country,
    COUNT(*)                                    AS txn_count,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)   AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 4) AS fraud_rate_pct
FROM transactions
GROUP BY txn_country
ORDER BY fraud_rate_pct DESC;

-- 5b. Cross-border transactions: frequent travelers vs. non-travelers
-- (this is the "realistic false positive" check -- legitimate travelers
-- should NOT show an elevated fraud rate on cross-border transactions,
-- unlike non-travelers doing the same thing)
SELECT
    c.is_frequent_traveler,
    t.is_cross_border,
    COUNT(*) AS txn_count,
    SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) / COUNT(*), 4) AS fraud_rate_pct
FROM transactions t
JOIN customers c ON c.customer_id = t.customer_id
GROUP BY c.is_frequent_traveler, t.is_cross_border
ORDER BY fraud_rate_pct DESC;

-- 5c. Large single-transaction location jumps (possible impossible-travel /
-- account-takeover signal)
SELECT
    transaction_id,
    customer_id,
    timestamp,
    amount,
    distance_from_prev_km,
    is_fraud
FROM transactions
WHERE distance_from_prev_km > 8000
ORDER BY distance_from_prev_km DESC
LIMIT 50;

-- 5d. Historical investigation queue performance (rule-based baseline,
-- pre-ML) -- precision/recall of the naive heuristic system
SELECT
    i.outcome,
    COUNT(*) AS n,
    ROUND(AVG(i.resolution_time_hours), 1) AS avg_resolution_hours
FROM investigations i
GROUP BY i.outcome
ORDER BY n DESC;

-- 5e. Rule-based investigation precision: of flagged cases, how many were
-- actually fraud? (baseline to beat with the ML expected-loss ranking)
SELECT
    COUNT(*)                                       AS total_investigated,
    SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END)    AS true_fraud_flagged,
    ROUND(100.0 * SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) / COUNT(*), 2) AS precision_pct
FROM investigations i
JOIN transactions t ON t.transaction_id = i.transaction_id;
