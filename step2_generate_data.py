"""
Step 2 — Generate Realistic Messy Sales Data
Produces 1000 rows of sales data with intentional nulls, duplicates,
and inconsistencies — mirroring real-world raw data scenarios.
Output: data/raw_sales_data.csv
"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_ROWS = 1000
NULL_RATE = 0.05          # 5 % of values become null
DUPLICATE_ROWS = 30       # exact duplicate rows injected

REGIONS = ["North", "South", "East", "West", "Central"]
PRODUCTS = [
    "Laptop", "Monitor", "Keyboard", "Mouse", "Headphones",
    "Webcam", "Desk Chair", "USB Hub", "Printer", "Tablet",
]
SALES_REPS = [fake.name() for _ in range(20)]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "PayPal", "Bank Transfer", "Cash"]
ORDER_STATUSES = ["Completed", "Pending", "Cancelled", "Refunded"]

# Realistic price ranges per product (min, max)
PRODUCT_PRICES = {
    "Laptop": (600, 2500),
    "Monitor": (150, 900),
    "Keyboard": (20, 250),
    "Mouse": (10, 150),
    "Headphones": (30, 400),
    "Webcam": (40, 300),
    "Desk Chair": (100, 800),
    "USB Hub": (15, 80),
    "Printer": (80, 600),
    "Tablet": (200, 1200),
}


# ── Helper functions ──────────────────────────────────────────────────────────

def random_date(start: datetime, end: datetime) -> str:
    delta = end - start
    random_days = random.randint(0, delta.days)
    return (start + timedelta(days=random_days)).strftime("%Y-%m-%d")


def inject_nulls(df: pd.DataFrame, rate: float) -> pd.DataFrame:
    """Randomly set ~rate fraction of cells to NaN (excluding order_id)."""
    nullable_cols = [c for c in df.columns if c != "order_id"]
    for col in nullable_cols:
        null_mask = np.random.random(len(df)) < rate
        df.loc[null_mask, col] = np.nan
    return df


def inject_duplicates(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Append n exact duplicate rows sampled from existing rows."""
    dupes = df.sample(n=n, random_state=SEED)
    return pd.concat([df, dupes], ignore_index=True)


def inject_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    """Add realistic data quality issues."""
    # Mixed-case region names
    mixed_case_idx = df.sample(frac=0.03, random_state=SEED).index
    df.loc[mixed_case_idx, "region"] = df.loc[mixed_case_idx, "region"].str.lower()

    # Negative quantities (data entry errors)
    neg_qty_idx = df.sample(frac=0.02, random_state=SEED + 1).index
    df.loc[neg_qty_idx, "quantity"] = df.loc[neg_qty_idx, "quantity"] * -1

    # Future dates (impossible order dates)
    future_idx = df.sample(frac=0.01, random_state=SEED + 2).index
    df.loc[future_idx, "order_date"] = "2027-01-15"

    # Whitespace in product names
    ws_idx = df.sample(frac=0.02, random_state=SEED + 3).index
    df.loc[ws_idx, "product"] = " " + df.loc[ws_idx, "product"] + " "

    return df


# ── Data generation ───────────────────────────────────────────────────────────

def generate_sales_data() -> pd.DataFrame:
    print("=" * 60)
    print("  Sales Data Pipeline — Step 2: Data Generation")
    print("=" * 60)

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)

    records = []
    for i in range(1, NUM_ROWS + 1):
        product = random.choice(PRODUCTS)
        price_min, price_max = PRODUCT_PRICES[product]
        unit_price = round(random.uniform(price_min, price_max), 2)
        quantity = random.randint(1, 20)
        discount = round(random.uniform(0, 0.30), 2)   # 0–30 % discount
        revenue = round(unit_price * quantity * (1 - discount), 2)

        records.append({
            "order_id": f"ORD-{i:05d}",
            "order_date": random_date(start_date, end_date),
            "region": random.choice(REGIONS),
            "product": product,
            "category": "Electronics",          # single category for simplicity
            "sales_rep": random.choice(SALES_REPS),
            "quantity": quantity,
            "unit_price": unit_price,
            "discount": discount,
            "revenue": revenue,
            "payment_method": random.choice(PAYMENT_METHODS),
            "order_status": random.choice(ORDER_STATUSES),
            "customer_id": f"CUST-{random.randint(1, 300):04d}",
        })

    df = pd.DataFrame(records)

    # Inject messiness
    print(f"\n📊 Base records generated : {len(df):,}")
    df = inject_nulls(df, NULL_RATE)
    print(f"🕳️  Nulls injected          : ~{int(len(df) * NULL_RATE * len(df.columns)):,} cells")
    df = inject_duplicates(df, DUPLICATE_ROWS)
    print(f"🔁 Duplicate rows added    : {DUPLICATE_ROWS}")
    df = inject_inconsistencies(df)
    print(f"⚠️  Inconsistencies added   : mixed case, negatives, future dates")
    print(f"\n📦 Final raw dataset shape : {df.shape}")

    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n💾 Saved to: {path}")


if __name__ == "__main__":
    OUTPUT_PATH = "data/raw_sales_data.csv"
    df = generate_sales_data()
    save_data(df, OUTPUT_PATH)

    print("\n📋 Sample rows:")
    print(df.head(5).to_string(index=False))
    print("\n📈 Null counts per column:")
    print(df.isnull().sum().to_string())
