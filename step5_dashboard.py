"""
Step 5 — 4-Chart Analytics Dashboard
Visualises the cleaned sales data with:
  Chart 1 — Revenue by Region (horizontal bar)
  Chart 2 — Monthly Revenue Trend (line + rolling average)
  Chart 3 — Product Revenue Share (donut chart)
  Chart 4 — Order Value Distribution (histogram + KDE)
Output: outputs/sales_dashboard.png
"""

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
CLEAN_PATH = "data/cleaned_sales_data.csv"
AGG_REGION = "data/aggregations/region_agg.csv"
AGG_MONTHLY = "data/aggregations/monthly_agg.csv"
AGG_PRODUCT = "data/aggregations/product_agg.csv"
OUTPUT_DIR = "outputs"
OUTPUT_FILE = f"{OUTPUT_DIR}/sales_dashboard.png"

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
BG_COLOR = "#F8F9FA"
GRID_COLOR = "#E0E0E0"
TEXT_COLOR = "#212121"

sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": BG_COLOR,
    "figure.facecolor": "white",
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
})


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(CLEAN_PATH, parse_dates=["order_date"])

    # Load or recompute aggregations
    try:
        region_df = pd.read_csv(AGG_REGION)
        monthly_df = pd.read_csv(AGG_MONTHLY)
        product_df = pd.read_csv(AGG_PRODUCT)
    except FileNotFoundError:
        print("⚠️  Aggregation files not found — recomputing from cleaned data.")
        region_df = (
            df.groupby("region")
              .agg(total_revenue=("revenue", "sum"), total_orders=("order_id", "count"))
              .round(2).reset_index()
              .sort_values("total_revenue", ascending=False)
        )
        monthly_df = (
            df.groupby("month_year")
              .agg(total_revenue=("revenue", "sum"), total_orders=("order_id", "count"))
              .round(2).reset_index().sort_values("month_year")
        )
        monthly_df["rolling_3m_revenue"] = (
            monthly_df["total_revenue"].rolling(3, min_periods=1).mean().round(2)
        )
        product_df = (
            df.groupby("product")
              .agg(total_revenue=("revenue", "sum"))
              .round(2).reset_index()
              .sort_values("total_revenue", ascending=False)
        )

    return df, region_df, monthly_df, product_df


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — Revenue by Region (horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────

def chart_revenue_by_region(ax, region_df: pd.DataFrame) -> None:
    region_df = region_df.sort_values("total_revenue")
    colors = sns.color_palette(PALETTE, len(region_df))

    bars = ax.barh(
        region_df["region"],
        region_df["total_revenue"],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        height=0.6,
    )

    # Value labels
    for bar, val in zip(bars, region_df["total_revenue"]):
        ax.text(
            bar.get_width() + bar.get_width() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"${val:,.0f}",
            va="center", ha="left", fontsize=9, color=TEXT_COLOR,
        )

    ax.set_title("Revenue by Region", pad=12)
    ax.set_xlabel("Total Revenue ($)")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
    ax.set_xlim(0, region_df["total_revenue"].max() * 1.18)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.7)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right"]].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — Monthly Revenue Trend
# ─────────────────────────────────────────────────────────────────────────────

def chart_monthly_trend(ax, monthly_df: pd.DataFrame) -> None:
    x = range(len(monthly_df))
    labels = monthly_df["month_year"].tolist()

    # Bar chart for monthly revenue
    ax.bar(x, monthly_df["total_revenue"], color=PALETTE[0], alpha=0.55,
           label="Monthly Revenue", zorder=2)

    # Rolling average line
    if "rolling_3m_revenue" in monthly_df.columns:
        ax.plot(x, monthly_df["rolling_3m_revenue"], color=PALETTE[3],
                linewidth=2.2, marker="o", markersize=4,
                label="3-Month Rolling Avg", zorder=3)

    ax.set_title("Monthly Revenue Trend", pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue ($)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y/1e3:.0f}K"))
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — Product Revenue Share (donut)
# ─────────────────────────────────────────────────────────────────────────────

def chart_product_share(ax, product_df: pd.DataFrame) -> None:
    # Show top 6 products, group rest as "Other"
    top_n = 6
    top = product_df.head(top_n).copy()
    other_rev = product_df.iloc[top_n:]["total_revenue"].sum()
    if other_rev > 0:
        other_row = pd.DataFrame([{"product": "Other", "total_revenue": other_rev}])
        top = pd.concat([top, other_row], ignore_index=True)

    colors = sns.color_palette(PALETTE + ["#607D8B", "#795548"], len(top))
    wedge_props = {"width": 0.45, "edgecolor": "white", "linewidth": 2}

    wedges, texts, autotexts = ax.pie(
        top["total_revenue"],
        labels=top["product"],
        autopct="%1.1f%%",
        colors=colors,
        wedgeprops=wedge_props,
        startangle=90,
        pctdistance=0.75,
    )

    for text in texts:
        text.set_fontsize(9)
    for autotext in autotexts:
        autotext.set_fontsize(8)
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    # Centre label
    total = product_df["total_revenue"].sum()
    ax.text(0, 0, f"${total/1e3:.0f}K\nTotal",
            ha="center", va="center", fontsize=10, fontweight="bold", color=TEXT_COLOR)

    ax.set_title("Product Revenue Share", pad=12)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 — Order Value Distribution
# ─────────────────────────────────────────────────────────────────────────────

def chart_order_distribution(ax, df: pd.DataFrame) -> None:
    revenue = df["revenue"].dropna()

    # Histogram
    n, bins, patches = ax.hist(
        revenue, bins=40, color=PALETTE[1], alpha=0.65,
        edgecolor="white", linewidth=0.5, density=True, label="Frequency",
    )

    # KDE overlay
    from scipy.stats import gaussian_kde  # optional; fallback if unavailable
    try:
        kde = gaussian_kde(revenue)
        x_range = np.linspace(revenue.min(), revenue.max(), 300)
        ax.plot(x_range, kde(x_range), color=PALETTE[3], linewidth=2.2, label="KDE")
    except Exception:
        pass  # skip KDE if scipy not available

    # Percentile lines
    for pct, label, color in [
        (0.25, "Q1", PALETTE[2]),
        (0.50, "Median", PALETTE[3]),
        (0.75, "Q3", PALETTE[4]),
    ]:
        val = revenue.quantile(pct)
        ax.axvline(val, color=color, linestyle="--", linewidth=1.4, alpha=0.85)
        ax.text(val, ax.get_ylim()[1] * 0.95, f" {label}\n ${val:,.0f}",
                color=color, fontsize=8, va="top")

    ax.set_title("Order Value Distribution", pad=12)
    ax.set_xlabel("Revenue per Order ($)")
    ax.set_ylabel("Density")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard() -> None:
    print("=" * 60)
    print("  Sales Data Pipeline — Step 5: Dashboard")
    print("=" * 60)

    df, region_df, monthly_df, product_df = load_data()
    print(f"\n✅ Data loaded: {len(df):,} rows")

    fig = plt.figure(figsize=(18, 12), facecolor="white")
    fig.suptitle(
        "Sales Analytics Dashboard",
        fontsize=18, fontweight="bold", color=TEXT_COLOR, y=0.98,
    )

    # 2×2 grid with spacing
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32,
                          left=0.07, right=0.97, top=0.93, bottom=0.08)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    chart_revenue_by_region(ax1, region_df)
    chart_monthly_trend(ax2, monthly_df)
    chart_product_share(ax3, product_df)
    chart_order_distribution(ax4, df)

    # Footer
    fig.text(
        0.5, 0.01,
        "Sales Data Pipeline  |  Data Engineering Portfolio Project",
        ha="center", fontsize=9, color="#9E9E9E",
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\n💾 Dashboard saved → {OUTPUT_FILE}")
    plt.show()
    print("\n✅ Step 5 complete!")


if __name__ == "__main__":
    build_dashboard()
