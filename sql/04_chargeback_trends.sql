-- Chargeback trends & loss analysis
-- Answers: how much money are we actually losing, and how does that compare
-- to raw fraud incidence?

-- 4a. Monthly chargeback volume and value, fraud-related vs. non-fraud disputes
SELECT
    STRFTIME('%Y-%m', cb.chargeback_date)              AS month,
    cb.is_fraud_related,
    COUNT(*)                                            AS chargeback_count,
    ROUND(SUM(cb.chargeback_amount), 2)                 AS chargeback_value
FROM chargebacks cb
GROUP BY month, cb.is_fraud_related
ORDER BY month, cb.is_fraud_related DESC;

-- 4b. Chargeback reason breakdown (fraud vs. legitimate dispute reasons)
SELECT
    cb.chargeback_reason,
    cb.is_fraud_related,
    COUNT(*) AS chargeback_count,
    ROUND(SUM(cb.chargeback_amount), 2) AS chargeback_value,
    ROUND(AVG(cb.chargeback_amount), 2) AS avg_chargeback_value
FROM chargebacks cb
GROUP BY cb.chargeback_reason, cb.is_fraud_related
ORDER BY chargeback_value DESC;

-- 4c. Detection gap: how much realized fraud NEVER shows up as a chargeback
-- within the data window (i.e. what pure chargeback tracking would miss)
SELECT
    (SELECT ROUND(SUM(amount), 2) FROM transactions WHERE is_fraud = 1) AS total_true_fraud_value,
    (SELECT ROUND(SUM(cb.chargeback_amount), 2)
       FROM chargebacks cb
       JOIN transactions t ON t.transaction_id = cb.transaction_id
      WHERE t.is_fraud = 1)                                             AS fraud_value_charged_back,
    (SELECT ROUND(SUM(amount), 2) FROM transactions WHERE is_fraud = 1)
      - (SELECT ROUND(SUM(cb.chargeback_amount), 2)
           FROM chargebacks cb
           JOIN transactions t ON t.transaction_id = cb.transaction_id
          WHERE t.is_fraud = 1)                                         AS undetected_or_undisputed_fraud_value;

-- 4d. Average lag between transaction and chargeback filing, by reason
SELECT
    cb.chargeback_reason,
    ROUND(AVG(JULIANDAY(cb.chargeback_date) - JULIANDAY(t.timestamp)), 1) AS avg_lag_days,
    COUNT(*) AS n
FROM chargebacks cb
JOIN transactions t ON t.transaction_id = cb.transaction_id
GROUP BY cb.chargeback_reason
ORDER BY avg_lag_days;
