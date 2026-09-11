"""
Generates the core entity tables: customers, accounts, devices, merchants.

These are the "static" tables that transactions will later reference.
Risk-relevant attributes are seeded here (e.g. account_age_days,
customer risk_segment, merchant category risk) so that transaction-level
fraud injection has something causal to key off of.
"""

import numpy as np
import pandas as pd
from faker import Faker

from config import CFG, SEED, DEVICE_COUNT_WEIGHTS, MERCHANT_CATEGORIES, RAW_DATA_DIR

rng = np.random.default_rng(SEED)
fake = Faker()
Faker.seed(SEED)

COUNTRIES_HOME = ["US"] * 85 + ["CA"] * 6 + ["GB"] * 4 + ["AU"] * 3 + ["DE"] * 2
US_STATES = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI",
             "WA", "AZ", "MA", "TN", "IN", "MO", "CO", "MD", "WI", "MN"]


def generate_customers(n: int) -> pd.DataFrame:
    """
    Customer master table. risk_segment is a latent variable that later
    drives fraud probability -- NOT observable to the model directly,
    mirroring the fact that real fraud analysts don't get to see "ground
    truth intent", only behavioral signals.
    """
    customer_ids = np.arange(1, n + 1)
    signup_days_ago = rng.integers(30, 365 * 6, size=n)  # up to 6 years tenure
    account_age_days = signup_days_ago

    # Latent risk segment: most customers are low risk; a small minority
    # are inherently higher risk (e.g. thin-file, high velocity spenders).
    risk_segment = rng.choice(
        ["low", "medium", "high"], size=n, p=[0.85, 0.12, 0.03]
    )

    home_country = rng.choice(COUNTRIES_HOME, size=n)
    home_state = np.where(
        home_country == "US", rng.choice(US_STATES, size=n), "N/A"
    )

    avg_monthly_spend = np.round(
        rng.lognormal(mean=6.2, sigma=0.9, size=n), 2
    )  # lognormal -> realistic right-skewed spend distribution
    avg_monthly_spend = np.clip(avg_monthly_spend, 20, 50_000)

    is_frequent_traveler = rng.random(n) < 0.08  # explains legit geo anomalies later

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "signup_date_days_ago": signup_days_ago,
        "account_age_days": account_age_days,
        "risk_segment": risk_segment,
        "home_country": home_country,
        "home_state": home_state,
        "avg_monthly_spend": avg_monthly_spend,
        "is_frequent_traveler": is_frequent_traveler,
        "date_of_birth_year": rng.integers(1945, 2005, size=n),
    })
    return df


def generate_accounts(customers: pd.DataFrame) -> pd.DataFrame:
    """
    Most customers have exactly one primary account; a minority have
    multiple (checking + savings + credit), which matters for
    cross-account velocity features later.
    """
    rows = []
    account_id_counter = 1
    n_accounts_choice = rng.choice([1, 2, 3], size=len(customers), p=[0.7, 0.25, 0.05])

    for cust_id, age_days, n_accts in zip(
        customers["customer_id"], customers["account_age_days"], n_accounts_choice
    ):
        for i in range(n_accts):
            acct_type = "primary" if i == 0 else rng.choice(["savings", "credit"])
            rows.append({
                "account_id": account_id_counter,
                "customer_id": cust_id,
                "account_type": acct_type,
                "account_age_days": max(age_days - rng.integers(0, 30), 1),
                "account_status": "active",
            })
            account_id_counter += 1
    return pd.DataFrame(rows)


def generate_devices(customers: pd.DataFrame) -> pd.DataFrame:
    """
    Devices per customer. A small fraction of devices are *shared* across
    multiple customers -- a classic fraud ring / account-takeover signal
    (device_accounts_count feature downstream).
    """
    device_counts = rng.choice(
        list(DEVICE_COUNT_WEIGHTS.keys()),
        size=len(customers),
        p=list(DEVICE_COUNT_WEIGHTS.values()),
    )

    rows = []
    device_id_counter = 1
    device_pool_for_sharing = []  # device_ids eligible to be reused by another customer

    for cust_id, n_devices in zip(customers["customer_id"], device_counts):
        for _ in range(n_devices):
            # ~1.5% chance this "new" device is actually a reused/shared device
            if device_pool_for_sharing and rng.random() < 0.015:
                shared_device_id = rng.choice(device_pool_for_sharing)
                rows.append({
                    "device_id": shared_device_id,
                    "customer_id": cust_id,
                    "device_type": rng.choice(["mobile", "desktop", "tablet"], p=[0.65, 0.3, 0.05]),
                    "first_seen_days_ago": rng.integers(1, 1000),
                    "is_shared_device": True,
                })
            else:
                rows.append({
                    "device_id": device_id_counter,
                    "customer_id": cust_id,
                    "device_type": rng.choice(["mobile", "desktop", "tablet"], p=[0.65, 0.3, 0.05]),
                    "first_seen_days_ago": rng.integers(1, 1000),
                    "is_shared_device": False,
                })
                device_pool_for_sharing.append(device_id_counter)
                device_id_counter += 1

    return pd.DataFrame(rows)


def generate_merchants(n: int) -> pd.DataFrame:
    """
    Merchant master table with category-driven baseline risk, plus a
    small tail of "high risk" merchants within any category (e.g. a
    compromised or genuinely bad-actor merchant) to give merchant_risk_score
    real predictive signal.
    """
    categories = list(MERCHANT_CATEGORIES.keys())
    category_weights = np.array([1.0] * len(categories))
    category_weights = category_weights / category_weights.sum()

    merchant_category = rng.choice(categories, size=n, p=category_weights)
    base_risk = np.array([MERCHANT_CATEGORIES[c] for c in merchant_category])

    # merchant-specific noise on top of category baseline (some grocery
    # stores are still riskier than others, etc.)
    merchant_noise = rng.lognormal(mean=0, sigma=0.4, size=n)
    merchant_risk_score = np.clip(base_risk * merchant_noise, 0.05, 8.0)

    is_high_risk_outlier = rng.random(n) < 0.02
    merchant_risk_score = np.where(
        is_high_risk_outlier, merchant_risk_score * rng.uniform(2, 4, size=n), merchant_risk_score
    )

    countries = rng.choice(COUNTRIES_HOME, size=n)

    df = pd.DataFrame({
        "merchant_id": np.arange(1, n + 1),
        "merchant_category": merchant_category,
        "merchant_country": countries,
        "merchant_risk_score": np.round(merchant_risk_score, 3),
        "merchant_age_days": rng.integers(30, 3000, size=n),
        "is_high_risk_outlier": is_high_risk_outlier,
    })
    return df


if __name__ == "__main__":
    import os
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    print(f"Generating {CFG.n_customers:,} customers...")
    customers = generate_customers(CFG.n_customers)
    customers.to_parquet(RAW_DATA_DIR + "/customers.parquet", index=False)

    print("Generating accounts...")
    accounts = generate_accounts(customers)
    accounts.to_parquet(RAW_DATA_DIR + "/accounts.parquet", index=False)

    print("Generating devices...")
    devices = generate_devices(customers)
    devices.to_parquet(RAW_DATA_DIR + "/devices.parquet", index=False)

    print(f"Generating {CFG.n_merchants:,} merchants...")
    merchants = generate_merchants(CFG.n_merchants)
    merchants.to_parquet(RAW_DATA_DIR + "/merchants.parquet", index=False)

    print("Done.")
    print(f"  customers: {len(customers):,}")
    print(f"  accounts:  {len(accounts):,}")
    print(f"  devices:   {len(devices):,}")
    print(f"  merchants: {len(merchants):,}")
