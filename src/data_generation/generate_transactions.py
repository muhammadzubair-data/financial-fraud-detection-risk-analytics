"""
Generates the transactions table -- the heart of the dataset.

Design principles (see project brief):
  1. Fraud is CAUSALLY embedded via behavioral signals, not assigned at random.
  2. We deliberately include realistic FALSE POSITIVE cases (e.g. a genuine
     traveler transacting from a new country) so the classification problem
     is honestly hard, not a toy.
  3. Every transaction is generated in chronological order per customer, with
     running state (rolling averages, last device, last location, recent
     transaction counts) computed causally -- i.e. only from *prior*
     transactions -- so downstream feature engineering has no leakage traps
     baked into the raw data itself.
  4. We do NOT persist the internal "true fraud probability" to the output
     table. Only the realized transaction facts and the final binary label
     are written, mirroring what a real fraud team would actually have.

Output columns are intentionally "observable" -- the kind of thing you'd
find in a real transactions table -- while the *reasons* a transaction is
risky are implicit in the combination of fields (amount, device_id,
location, time-since-last, merchant_id), exactly as in production data.
"""

import numpy as np
import pandas as pd
from datetime import timedelta

from config import CFG, SEED, TARGET_FRAUD_RATE, RAW_DATA_DIR

rng = np.random.default_rng(SEED + 1)

# Simple lat/lon centroids for a handful of countries/regions, used to
# simulate "distance from previous transaction" as an actual geo signal.
COUNTRY_CENTROIDS = {
    "US": (39.8, -98.6), "CA": (56.1, -106.3), "GB": (55.4, -3.4),
    "AU": (-25.3, 133.8), "DE": (51.2, 10.5), "NG": (9.1, 8.7),
    "RU": (61.5, 105.3), "CN": (35.9, 104.2), "BR": (-14.2, -51.9),
}
FRAUD_HOTSPOT_COUNTRIES = ["NG", "RU", "CN"]  # elevated baseline risk if txn routes here
ALL_COUNTRIES = list(COUNTRY_CENTROIDS.keys())


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def simulate_customer_stream(customer, customer_devices, merchants_df, start_date, end_date):
    """
    Generates the full chronological transaction stream for ONE customer,
    with running state carried forward transaction-to-transaction.
    Returns a list of dicts (transaction rows).
    """
    cust_id = customer["customer_id"]
    base_spend = customer["avg_monthly_spend"]
    risk_segment = customer["risk_segment"]
    home_country = customer["home_country"]
    is_traveler = customer["is_frequent_traveler"]

    home_lat, home_lon = COUNTRY_CENTROIDS.get(home_country, COUNTRY_CENTROIDS["US"])

    n_days = (end_date - start_date).days
    # Expected number of transactions for this customer over the whole window
    avg_per_day = (CFG.avg_transactions_per_customer_per_year * CFG.years_of_history) / max(n_days, 1)
    risk_multiplier = {"low": 1.0, "medium": 1.3, "high": 1.8}[risk_segment]

    # Simulate day-by-day using a Poisson process for txn counts per day
    daily_counts = rng.poisson(avg_per_day, size=n_days)

    device_ids = list(customer_devices) if len(customer_devices) else [None]
    primary_device = rng.choice(device_ids)

    # Running state
    txn_amounts_history = []
    last_lat, last_lon = home_lat, home_lon
    last_device = primary_device
    recent_txn_times = []  # for velocity features (rolling window pruning)

    # --- Account takeover episode (rare, only for a subset of customers) ---
    takeover_day = None
    if rng.random() < 0.015 * risk_multiplier:
        takeover_day = rng.integers(int(n_days * 0.3), n_days)  # not at the very start
        takeover_duration = rng.integers(1, 4)  # days of anomalous behavior

    # --- Card-testing burst episode (rare) ---
    burst_day = None
    if rng.random() < 0.008 * risk_multiplier:
        burst_day = rng.integers(0, n_days)

    rows = []
    merchant_ids = merchants_df["merchant_id"].values
    merchant_risk = merchants_df.set_index("merchant_id")["merchant_risk_score"]
    merchant_country_map = merchants_df.set_index("merchant_id")["merchant_country"]

    for day_offset in range(n_days):
        n_today = daily_counts[day_offset]
        if n_today == 0:
            continue
        current_date = start_date + timedelta(days=int(day_offset))

        in_takeover_window = (
            takeover_day is not None and takeover_day <= day_offset < takeover_day + takeover_duration
        )
        in_burst_window = burst_day is not None and day_offset == burst_day

        if in_burst_window:
            n_today = max(n_today, rng.integers(6, 15))  # force a velocity spike

        for _ in range(n_today):
            fraud_signal_score = 0.0  # accumulates evidence; converted to probability later
            reason_codes = []

            merchant_id = rng.choice(merchant_ids)
            m_risk = merchant_risk.loc[merchant_id]
            m_country = merchant_country_map.loc[merchant_id]

            # --- amount ---
            personal_avg = np.mean(txn_amounts_history[-30:]) if txn_amounts_history else base_spend / 20
            personal_avg = max(personal_avg, 5)

            if in_takeover_window:
                amount = personal_avg * rng.uniform(4, 12)
                device_used = primary_device if rng.random() < 0.1 else f"new_{cust_id}_{day_offset}"
                fraud_signal_score += 5.5
                reason_codes.append("account_takeover_pattern")
            elif in_burst_window:
                amount = rng.uniform(1, 15)  # small card-testing amounts
                device_used = last_device
                fraud_signal_score += 4.2
                reason_codes.append("velocity_burst")
            else:
                amount = np.clip(rng.lognormal(mean=np.log(personal_avg), sigma=0.7), 1, 50_000)
                device_used = last_device if rng.random() < 0.9 else rng.choice(device_ids)

            amount_zscore = (amount - personal_avg) / max(np.std(txn_amounts_history[-30:]) if len(txn_amounts_history) > 3 else personal_avg * 0.5, 1)
            if amount_zscore > 4:
                fraud_signal_score += 2.0
                reason_codes.append("amount_far_above_baseline")

            # --- device ---
            new_device_flag = device_used != last_device
            if new_device_flag and not in_takeover_window:
                if rng.random() < 0.3:
                    fraud_signal_score += 0.8
                    reason_codes.append("new_device")

            # --- geography ---
            if in_takeover_window and rng.random() < 0.7:
                txn_country = rng.choice(FRAUD_HOTSPOT_COUNTRIES)
                fraud_signal_score += 2.0
                reason_codes.append("high_risk_geography")
            elif is_traveler and rng.random() < 0.12:
                # LEGITIMATE anomaly: real traveler, no other risk signal added
                txn_country = rng.choice([c for c in ALL_COUNTRIES if c != home_country])
            elif rng.random() < 0.015:
                txn_country = rng.choice(ALL_COUNTRIES)
                if txn_country in FRAUD_HOTSPOT_COUNTRIES:
                    fraud_signal_score += 1.4
                    reason_codes.append("unusual_geography")
            else:
                txn_country = home_country

            txn_lat, txn_lon = COUNTRY_CENTROIDS.get(txn_country, (home_lat, home_lon))
            # small jitter so it's not always exactly the centroid
            txn_lat += rng.normal(0, 1.5)
            txn_lon += rng.normal(0, 1.5)
            distance_km = haversine_km(last_lat, last_lon, txn_lat, txn_lon)

            # --- merchant risk contributes directly ---
            fraud_signal_score += 0.55 * np.log1p(m_risk)
            if m_risk > 2.5:
                reason_codes.append("high_risk_merchant")

            # --- velocity: recent txns in last hour/day (based on running list) ---
            txn_time = current_date + timedelta(seconds=int(rng.integers(0, 86400)))
            recent_txn_times.append(txn_time)
            recent_txn_times = [t for t in recent_txn_times if (txn_time - t).total_seconds() <= 86400]
            txns_last_24h = len(recent_txn_times)
            txns_last_1h = sum(1 for t in recent_txn_times if (txn_time - t).total_seconds() <= 3600)
            if txns_last_1h >= 5:
                fraud_signal_score += 1.6
                reason_codes.append("high_velocity")

            # --- failed attempts (declined-then-approved pattern) ---
            failed_attempts_24h = 0
            if fraud_signal_score > 1.5 and rng.random() < 0.4:
                failed_attempts_24h = rng.integers(1, 4)
                fraud_signal_score += 1.0
                reason_codes.append("repeated_failed_attempts")

            # base risk floor from segment + small baseline noise so not every
            # fraud case has an obvious signal (keeps some "unexplainable" fraud)
            # This floor is tuned so that a transaction with NO risk signals at
            # all sits far below the target rate, and only accumulated evidence
            # pushes probability up meaningfully -- avoiding "every transaction
            # is 5%+ fraud" degenerate calibration.
            base_floor = {"low": -6.1, "medium": -5.5, "high": -4.8}[risk_segment]
            logit = base_floor + fraud_signal_score + rng.normal(0, 0.35)
            fraud_probability = 1 / (1 + np.exp(-logit))

            is_fraud = rng.random() < fraud_probability

            rows.append({
                "customer_id": cust_id,
                "merchant_id": merchant_id,
                "device_id": device_used if isinstance(device_used, (int, np.integer)) else None,
                "device_id_raw": str(device_used),
                "timestamp": txn_time,
                "amount": round(float(amount), 2),
                "txn_country": txn_country,
                "txn_lat": round(float(txn_lat), 4),
                "txn_lon": round(float(txn_lon), 4),
                "distance_from_prev_km": round(float(distance_km), 2),
                "new_device_flag": bool(new_device_flag),
                "failed_attempts_24h": int(failed_attempts_24h),
                "is_fraud": bool(is_fraud),
                "_reason_codes": ",".join(reason_codes) if is_fraud else "",  # kept for validation/EDA only
            })

            txn_amounts_history.append(amount)
            last_lat, last_lon = txn_lat, txn_lon
            last_device = device_used

    return rows


if __name__ == "__main__":
    import os
    import time

    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    print("Loading entity tables...")
    customers = pd.read_parquet(RAW_DATA_DIR + "/customers.parquet")
    devices = pd.read_parquet(RAW_DATA_DIR + "/devices.parquet")
    merchants = pd.read_parquet(RAW_DATA_DIR + "/merchants.parquet")

    device_map = devices.groupby("customer_id")["device_id"].apply(list).to_dict()

    from config import END_DATE
    start_date = CFG.start_date
    end_date = END_DATE

    print(f"Generating transactions for {len(customers):,} customers "
          f"from {start_date.date()} to {end_date.date()}...")

    t0 = time.time()
    all_rows = []
    for i, customer in customers.iterrows():
        cust_devices = device_map.get(customer["customer_id"], [])
        rows = simulate_customer_stream(customer, cust_devices, merchants, start_date, end_date)
        all_rows.extend(rows)
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1:,}/{len(customers):,} customers "
                  f"({len(all_rows):,} txns so far, {elapsed:.1f}s elapsed)")

    txns = pd.DataFrame(all_rows)
    txns = txns.sort_values("timestamp").reset_index(drop=True)
    txns.insert(0, "transaction_id", np.arange(1, len(txns) + 1))

    fraud_rate = txns["is_fraud"].mean()
    print(f"\nGenerated {len(txns):,} transactions in {time.time()-t0:.1f}s")
    print(f"Realized fraud rate: {fraud_rate:.4%} (target was {TARGET_FRAUD_RATE:.4%})")

    txns.to_parquet(RAW_DATA_DIR + "/transactions.parquet", index=False)
    print("Saved to data/raw/transactions.parquet")
