"""
Step 3 — Pandas + NumPy Data Cleaning, Feature Engineering & Aggregation
This is the core of the project — covers ~70% of what interviewers test:
  • Null handling strategies
  • Deduplication
  • Type casting & date parsing
  • Outlier detection (IQR method)
  • Feature engineering (new columns derived from existing ones)
  • GroupBy aggregations
  • Pivot tables
  • Rolling window calculations
Output: data/cleaned_sales_data.csv, data/aggregated_sales.csv
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RAW_PATH = "data/raw_sales_data.csv"
CLEAN_PATH = "data/cleaned_sales_data.csv"
AGG_PATH = "data/aggregated_sales.csv"


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"✅ Loaded  : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Nulls   : {df.isnull().sum().sum():,} total")
    print(f"   Dupes   : {df.duplicated().sum():,} exact duplicates")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLEAN
# ─────────────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── Cleaning ─────────────────────────────────────────────")

    # 2a. Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"   Duplicates removed   : {before - len(df)}")

    # 2b. Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    # 2c. Standardise region to Title Case
    df["region"] = df["region"].str.title()

    # 2d. Parse order_date; coerce invalid dates (e.g. future) to NaT
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    today = pd.Timestamp("today").normalize()
    df.loc[df["order_date"] > today, "order_date"] = pd.NaT
    print(f"   Future dates nulled  : {df['order_date'].isna().sum()}")

    # 2e. Fix negative quantities — take absolute value
    neg_mask = df["quantity"] < 0
    df.loc[neg_mask, "quantity"] = df.loc[neg_mask, "quantity"].abs()
    print(f"   Negative qty fixed   : {neg_mask.sum()}")

    # 2f. Null imputation
    #   Numeric columns → median (robust to outliers)
    num_cols = ["quantity", "unit_price", "discount", "revenue"]
    for col in num_cols:
        median_val = df[col].median()
        nulls = df[col].isna().sum()
        df[col] = df[col].fillna(median_val)
        if nulls:
            print(f"   Imputed {col:<12} : {nulls} nulls → median ({median_val:.2f})")

    #   Categorical columns → mode
    cat_cols = ["region", "product", "sales_rep", "payment_method",
                "order_status", "customer_id"]
    for col in cat_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()[0]
            nulls = df[col].isna().sum()
            df[col] = df[col].fillna(mode_val)
            print(f"   Imputed {col:<12} : {nulls} nulls → mode ('{mode_val}')")

    #   Dates → forward fill after sorting
    df = df.sort_values("order_date")
    df["order_date"] = df["order_date"].ffill()

    # 2g. Outlier removal on revenue using IQR
    Q1 = df["revenue"].quantile(0.25)
    Q3 = df["revenue"].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    before = len(df)
    df = df[(df["revenue"] >= lower) & (df["revenue"] <= upper)]
    print(f"   Outliers removed     : {before - len(df)} (IQR method on revenue)")

    # 2h. Ensure correct dtypes
    df["quantity"] = df["quantity"].astype(int)
    df["discount"] = df["discount"].clip(0, 1)   # discount must be 0–100 %

    print(f"\n   ✅ Clean shape: {df.shape}")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── Feature Engineering ──────────────────────────────────")

    # Date-based features
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["quarter"] = df["order_date"].dt.quarter
    df["day_of_week"] = df["order_date"].dt.day_name()
    df["is_weekend"] = df["order_date"].dt.dayofweek >= 5
    df["month_year"] = df["order_date"].dt.to_period("M").astype(str)

    # Revenue-based features
    df["revenue_per_unit"] = (df["revenue"] / df["quantity"]).round(2)
    df["discount_amount"] = (df["unit_price"] * df["quantity"] * df["discount"]).round(2)
    df["gross_revenue"] = (df["unit_price"] * df["quantity"]).round(2)

    # Discount tier (binning)
    df["discount_tier"] = pd.cut(
        df["discount"],
        bins=[-0.001, 0.05, 0.15, 0.25, 1.0],
        labels=["None (<5%)", "Low (5-15%)", "Medium (15-25%)", "High (>25%)"],
    )

    # Revenue tier (quantile-based)
    df["revenue_tier"] = pd.qcut(
        df["revenue"],
        q=4,
        labels=["Low", "Medium", "High", "Premium"],
    )

    # Order value flag
    median_rev = df["revenue"].median()
    df["is_high_value"] = df["revenue"] > median_rev

    # Recency score: days since order (relative to dataset max date)
    max_date = df["order_date"].max()
    df["days_since_order"] = (max_date - df["order_date"]).dt.days

    print(f"   New columns added    : year, month, quarter, day_of_week,")
    print(f"                          is_weekend, month_year, revenue_per_unit,")
    print(f"                          discount_amount, gross_revenue,")
    print(f"                          discount_tier, revenue_tier,")
    print(f"                          is_high_value, days_since_order")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. AGGREGATIONS
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    print("\n── Aggregations ─────────────────────────────────────────")

    # 4a. Revenue by region
    region_agg = (
        df.groupby("region")
        .agg(
            total_revenue=("revenue", "sum"),
            avg_revenue=("revenue", "mean"),
            total_orders=("order_id", "count"),
            avg_discount=("discount", "mean"),
        )
        .round(2)
        .sort_values("total_revenue", ascending=False)
        .reset_index()
    )
    print(f"   Region aggregation   : {region_agg.shape}")

    # 4b. Monthly revenue trend
    monthly_agg = (
        df.groupby("month_year")
        .agg(
            total_revenue=("revenue", "sum"),
            total_orders=("order_id", "count"),
            avg_order_value=("revenue", "mean"),
        )
        .round(2)
        .reset_index()
        .sort_values("month_year")
    )
    # Rolling 3-month average
    monthly_agg["rolling_3m_revenue"] = (
        monthly_agg["total_revenue"].rolling(window=3, min_periods=1).mean().round(2)
    )
    print(f"   Monthly aggregation  : {monthly_agg.shape}")

    # 4c. Product performance
    product_agg = (
        df.groupby("product")
        .agg(
            total_revenue=("revenue", "sum"),
            total_units=("quantity", "sum"),
            avg_unit_price=("unit_price", "mean"),
            order_count=("order_id", "count"),
            avg_discount=("discount", "mean"),
        )
        .round(2)
        .sort_values("total_revenue", ascending=False)
        .reset_index()
    )
    product_agg["revenue_share_pct"] = (
        (product_agg["total_revenue"] / product_agg["total_revenue"].sum() * 100).round(2)
    )
    print(f"   Product aggregation  : {product_agg.shape}")

    # 4d. Sales rep leaderboard
    rep_agg = (
        df.groupby("sales_rep")
        .agg(
            total_revenue=("revenue", "sum"),
            order_count=("order_id", "count"),
            avg_revenue=("revenue", "mean"),
            unique_products=("product", "nunique"),
        )
        .round(2)
        .sort_values("total_revenue", ascending=False)
        .reset_index()
    )
    rep_agg["rank"] = rep_agg["total_revenue"].rank(ascending=False).astype(int)
    print(f"   Sales rep leaderboard: {rep_agg.shape}")

    # 4e. Pivot table — region × product revenue
    pivot = df.pivot_table(
        values="revenue",
        index="region",
        columns="product",
        aggfunc="sum",
        fill_value=0,
    ).round(2)
    print(f"   Pivot (region×product): {pivot.shape}")

    # 4f. Combined summary (region + month)
    summary = (
        df.groupby(["region", "month_year"])
        .agg(total_revenue=("revenue", "sum"), order_count=("order_id", "count"))
        .round(2)
        .reset_index()
    )

    return {
        "region": region_agg,
        "monthly": monthly_agg,
        "product": product_agg,
        "sales_rep": rep_agg,
        "pivot": pivot,
        "summary": summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame, aggs: dict[str, pd.DataFrame]) -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/aggregations", exist_ok=True)

    df.to_csv(CLEAN_PATH, index=False)
    print(f"\n💾 Cleaned data  → {CLEAN_PATH}")

    for name, agg_df in aggs.items():
        path = f"data/aggregations/{name}_agg.csv"
        agg_df.to_csv(path, index=True if name == "pivot" else False)
        print(f"💾 {name:<12} agg → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Sales Data Pipeline — Step 3: Pandas Pipeline")
    print("=" * 60)

    df_raw = load_data(RAW_PATH)
    df_clean = clean_data(df_raw)
    df_featured = engineer_features(df_clean)
    aggregations = aggregate_data(df_featured)
    save_outputs(df_featured, aggregations)

    print("\n📋 Cleaned data sample:")
    print(df_featured[["order_id", "order_date", "region", "product",
                        "revenue", "revenue_tier", "month_year"]].head(5).to_string(index=False))

    print("\n📊 Region summary:")
    print(aggregations["region"].to_string(index=False))

    print("\n✅ Step 3 complete!")
