"""
Central configuration for the fraud detection synthetic data generator.

Two scales are supported:
  - DEV: fast to generate, fits in memory easily, good for iterating on
         features/models/dashboards locally.
  - FULL: the "portfolio" scale described in the project brief
          (1-2M customers, 5-10M transactions, 2-3 years of history).

Everything is seeded for reproducibility. Change SCALE to switch sizes;
all downstream scripts read from this file so the two modes stay consistent.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

# ---- pick your scale here -------------------------------------------------
SCALE = "dev"   # "dev" or "full"
SEED = 42
# -----------------------------------------------------------------------------

END_DATE = datetime(2025, 12, 31)


@dataclass(frozen=True)
class ScaleConfig:
    n_customers: int
    n_merchants: int
    years_of_history: int
    avg_transactions_per_customer_per_year: int

    @property
    def start_date(self) -> datetime:
        return END_DATE - timedelta(days=365 * self.years_of_history)

    @property
    def approx_n_transactions(self) -> int:
        return self.n_customers * self.avg_transactions_per_customer_per_year * self.years_of_history


SCALE_CONFIGS = {
    # Small enough to generate + model on a laptop in minutes, same logic as FULL.
    "dev": ScaleConfig(
        n_customers=6_000,
        n_merchants=800,
        years_of_history=2,
        avg_transactions_per_customer_per_year=60,
    ),
    # The scale described in the project brief.
    "full": ScaleConfig(
        n_customers=1_500_000,
        n_merchants=40_000,
        years_of_history=3,
        avg_transactions_per_customer_per_year=60,
    ),
}

CFG = SCALE_CONFIGS[SCALE]

# Target overall fraud rate (fraction of transactions that are truly fraudulent).
# Kept realistic and low on purpose -- this is what makes the ML problem hard.
TARGET_FRAUD_RATE = 0.006  # ~0.6%

# Devices per customer (most customers use 1-2 devices; some share devices --
# useful signal for device-risk features).
DEVICE_COUNT_WEIGHTS = {1: 0.60, 2: 0.30, 3: 0.08, 4: 0.02}

# Merchant category codes and their baseline risk multiplier.
# Higher multiplier = inherently more fraud-prone category (card-testing,
# resale of stolen goods, anonymous digital goods, etc.)
MERCHANT_CATEGORIES = {
    "grocery": 0.3,
    "restaurant": 0.4,
    "gas_station": 0.5,
    "utilities": 0.1,
    "healthcare": 0.2,
    "clothing_retail": 0.7,
    "electronics": 1.8,
    "travel": 1.3,
    "digital_goods": 2.5,
    "gift_cards": 3.0,
    "money_transfer": 3.5,
    "gambling": 2.8,
    "jewelry": 2.2,
    "subscription": 0.6,
    "home_improvement": 0.4,
}

import os as _os

# Repo root = two levels up from this file (src/data_generation/config.py -> repo root)
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
RAW_DATA_DIR = _os.path.join(REPO_ROOT, "data", "raw")
DATA_DICTIONARY_PATH = _os.path.join(REPO_ROOT, "docs", "data_dictionary.md")
