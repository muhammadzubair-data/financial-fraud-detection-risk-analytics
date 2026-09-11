-- Transaction velocity & rolling-window fraud signals
-- Answers: which customers/devices show abnormal transaction bursts?

-- 1a. Daily transaction volume, value, and fraud rate (management trend view)
SELECT
    DATE(timestamp)                                        AS txn_date,
    COUNT(*)                                                AS txn_count,
    ROUND(SUM(amount), 2)                                   AS total_value,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)               AS fraud_count,
    ROUND(SUM(CASE WHEN is_fraud THEN amount ELSE 0 END), 2) AS fraud_value,
    ROUND(100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 4) AS fraud_rate_pct
FROM transactions
GROUP BY DATE(timestamp)
ORDER BY txn_date;

-- 1b. Customers with the highest same-day transaction velocity
-- (flags card-testing / account-takeover bursts)
SELECT
    customer_id,
    DATE(timestamp)              AS txn_date,
    COUNT(*)                     AS txns_that_day,
    ROUND(SUM(amount), 2)        AS value_that_day,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud_txns_that_day
FROM transactions
GROUP BY customer_id, DATE(timestamp)
HAVING COUNT(*) >= 6
ORDER BY txns_that_day DESC
LIMIT 50;

-- 1c. Rolling 7-day fraud rate trend (window function), to catch
-- deterioration/improvement over time rather than a single daily spike
SELECT
    txn_date,
    txn_count,
    fraud_count,
    ROUND(
        100.0 * SUM(fraud_count) OVER (ORDER BY txn_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
        / NULLIF(SUM(txn_count) OVER (ORDER BY txn_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 0),
        4
    ) AS rolling_7d_fraud_rate_pct
FROM (
    SELECT
        DATE(timestamp) AS txn_date,
        COUNT(*) AS txn_count,
        SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud_count
    FROM transactions
    GROUP BY DATE(timestamp)
) daily
ORDER BY txn_date;

-- 1d. Highest-velocity single-hour bursts across the whole dataset
-- (surfaces card-testing episodes for investigation)
SELECT
    customer_id,
    STRFTIME('%Y-%m-%d %H:00', timestamp) AS hour_bucket,
    COUNT(*) AS txns_in_hour,
    ROUND(SUM(amount), 2) AS value_in_hour,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud_in_hour
FROM transactions
GROUP BY customer_id, hour_bucket
HAVING COUNT(*) >= 5
ORDER BY txns_in_hour DESC
LIMIT 50;
