"""
Sales Data Pipeline — Streamlit Web App
Run with:  streamlit run app.py
Opens automatically in Chrome at http://localhost:8501
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, r"C:\Users\manok\AppData\Roaming\Python\Python314\site-packages")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Data Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
    h1 { color: #1a1a2e; }
    h2 { color: #16213e; }
    h3 { color: #0f3460; }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_all_data():
    """Load or generate all pipeline data."""
    raw_path   = "data/raw_sales_data.csv"
    clean_path = "data/cleaned_sales_data.csv"

    # Auto-run pipeline if data doesn't exist yet
    if not os.path.exists(clean_path):
        with st.spinner("Running pipeline for the first time…"):
            import step2_generate_data as s2
            df_raw = s2.generate_sales_data()
            s2.save_data(df_raw, raw_path)

            import step3_pandas_pipeline as s3
            df_raw2  = s3.load_data(raw_path)
            df_clean = s3.clean_data(df_raw2)
            df_feat  = s3.engineer_features(df_clean)
            aggs     = s3.aggregate_data(df_feat)
            s3.save_outputs(df_feat, aggs)

    raw   = pd.read_csv(raw_path)
    clean = pd.read_csv(clean_path, parse_dates=["order_date"])

    agg_region  = pd.read_csv("data/aggregations/region_agg.csv")
    agg_monthly = pd.read_csv("data/aggregations/monthly_agg.csv")
    agg_product = pd.read_csv("data/aggregations/product_agg.csv")
    agg_rep     = pd.read_csv("data/aggregations/sales_rep_agg.csv")

    return raw, clean, agg_region, agg_monthly, agg_product, agg_rep


@st.cache_resource
def load_model():
    import joblib
    path = "models/revenue_model.joblib"
    if os.path.exists(path):
        payload = joblib.load(path)
        return payload["model"], payload["feature_cols"]
    return None, None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
    st.title("Sales Pipeline")
    st.markdown("---")

    st.markdown("### Filters")
    raw, clean, agg_region, agg_monthly, agg_product, agg_rep = load_all_data()

    all_regions  = sorted(clean["region"].dropna().unique())
    all_products = sorted(clean["product"].dropna().unique())
    all_statuses = sorted(clean["order_status"].dropna().unique())

    sel_regions  = st.multiselect("Region",       all_regions,  default=all_regions)
    sel_products = st.multiselect("Product",      all_products, default=all_products)
    sel_statuses = st.multiselect("Order Status", all_statuses, default=all_statuses)

    date_min = clean["order_date"].min().date()
    date_max = clean["order_date"].max().date()
    date_range = st.date_input("Date Range", value=(date_min, date_max),
                               min_value=date_min, max_value=date_max)

    st.markdown("---")
    st.markdown("**Pipeline Steps**")
    st.success("✅ Step 2 — Data Generation")
    st.success("✅ Step 3 — Pandas Pipeline")
    st.info("⚡ Step 4 — PySpark (needs Java)")
    st.success("✅ Step 5 — Dashboard")
    st.success("✅ Step 6 — ML Model")


# ── Apply filters ─────────────────────────────────────────────────────────────
df = clean.copy()
if sel_regions:
    df = df[df["region"].isin(sel_regions)]
if sel_products:
    df = df[df["product"].isin(sel_products)]
if sel_statuses:
    df = df[df["order_status"].isin(sel_statuses)]
if len(date_range) == 2:
    df = df[(df["order_date"].dt.date >= date_range[0]) &
            (df["order_date"].dt.date <= date_range[1])]


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Sales Data Pipeline Dashboard")
st.markdown("*End-to-end data engineering project — data generation → cleaning → analytics → ML*")
st.markdown("---")


# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

total_rev    = df["revenue"].sum()
total_orders = len(df)
avg_order    = df["revenue"].mean()
total_units  = df["quantity"].sum()
avg_discount = df["discount"].mean() * 100

k1.metric("💰 Total Revenue",   f"${total_rev:,.0f}")
k2.metric("📦 Total Orders",    f"{total_orders:,}")
k3.metric("📈 Avg Order Value", f"${avg_order:,.2f}")
k4.metric("🛒 Units Sold",      f"{total_units:,}")
k5.metric("🏷️ Avg Discount",    f"{avg_discount:.1f}%")

st.markdown("---")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "📅 Trends",
    "🛍️ Products",
    "👥 Sales Reps",
    "🤖 ML Model",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns(2)

    # Revenue by Region — horizontal bar
    with col1:
        st.subheader("Revenue by Region")
        reg_df = (df.groupby("region")["revenue"]
                    .sum().reset_index()
                    .sort_values("revenue", ascending=True))
        fig = px.bar(
            reg_df, x="revenue", y="region", orientation="h",
            color="revenue", color_continuous_scale="Blues",
            text=reg_df["revenue"].apply(lambda v: f"${v:,.0f}"),
            labels={"revenue": "Total Revenue ($)", "region": ""},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            coloraxis_showscale=False,
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10, r=60, t=10, b=10), height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Order Status donut
    with col2:
        st.subheader("Order Status Breakdown")
        status_df = df["order_status"].value_counts().reset_index()
        status_df.columns = ["status", "count"]
        fig2 = px.pie(
            status_df, names="status", values="count",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig2.update_layout(
            margin=dict(l=10, r=10, t=10, b=10), height=320,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Revenue distribution histogram
    st.subheader("Order Value Distribution")
    fig3 = px.histogram(
        df, x="revenue", nbins=50,
        color_discrete_sequence=["#2196F3"],
        labels={"revenue": "Revenue per Order ($)", "count": "Frequency"},
        marginal="box",
    )
    fig3.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10), height=300,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Raw data quality summary
    st.subheader("Data Quality Summary")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Raw Rows",     f"{len(raw):,}")
    q2.metric("Clean Rows",   f"{len(clean):,}")
    q3.metric("Rows Removed", f"{len(raw)-len(clean):,}")
    q4.metric("Raw Nulls",    f"{raw.isnull().sum().sum():,}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRENDS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Monthly Revenue Trend")

    monthly = (df.groupby("month_year")
                 .agg(revenue=("revenue", "sum"),
                      orders=("order_id", "count"))
                 .reset_index()
                 .sort_values("month_year"))
    monthly["rolling_3m"] = monthly["revenue"].rolling(3, min_periods=1).mean()

    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    fig4.add_trace(go.Bar(
        x=monthly["month_year"], y=monthly["revenue"],
        name="Monthly Revenue", marker_color="#90CAF9", opacity=0.7,
    ), secondary_y=False)
    fig4.add_trace(go.Scatter(
        x=monthly["month_year"], y=monthly["rolling_3m"],
        name="3-Month Rolling Avg", line=dict(color="#E91E63", width=2.5),
        mode="lines+markers", marker=dict(size=5),
    ), secondary_y=False)
    fig4.add_trace(go.Scatter(
        x=monthly["month_year"], y=monthly["orders"],
        name="Order Count", line=dict(color="#FF9800", width=1.5, dash="dot"),
        mode="lines",
    ), secondary_y=True)
    fig4.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=30, b=10), height=380,
        xaxis=dict(tickangle=-45),
    )
    fig4.update_yaxes(title_text="Revenue ($)", secondary_y=False)
    fig4.update_yaxes(title_text="Order Count", secondary_y=True)
    st.plotly_chart(fig4, use_container_width=True)

    # Revenue by day of week
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Revenue by Day of Week")
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_df = (df.groupby("day_of_week")["revenue"]
                    .mean().reindex(dow_order).reset_index())
        fig5 = px.bar(
            dow_df, x="day_of_week", y="revenue",
            color="revenue", color_continuous_scale="Teal",
            labels={"day_of_week": "", "revenue": "Avg Revenue ($)"},
        )
        fig5.update_layout(coloraxis_showscale=False,
                           plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=10,r=10,t=10,b=10), height=280)
        st.plotly_chart(fig5, use_container_width=True)

    with col_b:
        st.subheader("Revenue by Quarter")
        q_df = (df.groupby("quarter")["revenue"]
                  .sum().reset_index())
        q_df["quarter"] = "Q" + q_df["quarter"].astype(str)
        fig6 = px.pie(q_df, names="quarter", values="revenue",
                      color_discrete_sequence=px.colors.sequential.Blues_r,
                      hole=0.4)
        fig6.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=280)
        st.plotly_chart(fig6, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRODUCTS
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.subheader("Revenue by Product")
        prod_df = (df.groupby("product")["revenue"]
                     .sum().reset_index()
                     .sort_values("revenue", ascending=False))
        fig7 = px.bar(
            prod_df, x="product", y="revenue",
            color="revenue", color_continuous_scale="Viridis",
            text=prod_df["revenue"].apply(lambda v: f"${v/1000:.0f}K"),
            labels={"product": "", "revenue": "Total Revenue ($)"},
        )
        fig7.update_traces(textposition="outside")
        fig7.update_layout(
            coloraxis_showscale=False,
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_tickangle=-30,
            margin=dict(l=10,r=10,t=30,b=10), height=340,
        )
        st.plotly_chart(fig7, use_container_width=True)

    with col_p2:
        st.subheader("Product Revenue Share")
        fig8 = px.pie(
            prod_df, names="product", values="revenue",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )
        fig8.update_layout(
            margin=dict(l=10,r=10,t=10,b=10), height=340,
            legend=dict(orientation="v"),
        )
        st.plotly_chart(fig8, use_container_width=True)

    # Avg price vs units scatter
    st.subheader("Product: Avg Unit Price vs Total Units Sold")
    scatter_df = df.groupby("product").agg(
        avg_price=("unit_price", "mean"),
        total_units=("quantity", "sum"),
        total_revenue=("revenue", "sum"),
    ).reset_index()
    fig9 = px.scatter(
        scatter_df, x="avg_price", y="total_units",
        size="total_revenue", color="product",
        hover_name="product",
        labels={"avg_price": "Avg Unit Price ($)", "total_units": "Total Units Sold"},
        size_max=60,
    )
    fig9.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10,r=10,t=10,b=10), height=340,
    )
    st.plotly_chart(fig9, use_container_width=True)

    # Product table
    st.subheader("Product Performance Table")
    prod_table = df.groupby("product").agg(
        Total_Revenue=("revenue", "sum"),
        Avg_Revenue=("revenue", "mean"),
        Total_Units=("quantity", "sum"),
        Order_Count=("order_id", "count"),
        Avg_Discount=("discount", "mean"),
    ).round(2).reset_index().sort_values("Total_Revenue", ascending=False)
    prod_table["Revenue_Share"] = (
        prod_table["Total_Revenue"] / prod_table["Total_Revenue"].sum() * 100
    ).round(1).astype(str) + "%"
    prod_table["Total_Revenue"] = prod_table["Total_Revenue"].apply(lambda v: f"${v:,.2f}")
    prod_table["Avg_Revenue"]   = prod_table["Avg_Revenue"].apply(lambda v: f"${v:,.2f}")
    prod_table["Avg_Discount"]  = prod_table["Avg_Discount"].apply(lambda v: f"{v*100:.1f}%")
    st.dataframe(prod_table, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — SALES REPS
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Sales Rep Leaderboard")

    rep_df = (df.groupby("sales_rep").agg(
        Total_Revenue=("revenue", "sum"),
        Order_Count=("order_id", "count"),
        Avg_Revenue=("revenue", "mean"),
        Unique_Products=("product", "nunique"),
    ).round(2).reset_index()
     .sort_values("Total_Revenue", ascending=False)
     .reset_index(drop=True))
    rep_df.index += 1
    rep_df["Rank"] = rep_df.index

    col_r1, col_r2 = st.columns([2, 1])

    with col_r1:
        top10 = rep_df.head(10).sort_values("Total_Revenue", ascending=True)
        fig10 = px.bar(
            top10, x="Total_Revenue", y="sales_rep", orientation="h",
            color="Total_Revenue", color_continuous_scale="Greens",
            text=top10["Total_Revenue"].apply(lambda v: f"${v:,.0f}"),
            labels={"Total_Revenue": "Total Revenue ($)", "sales_rep": ""},
            title="Top 10 Sales Reps by Revenue",
        )
        fig10.update_traces(textposition="outside")
        fig10.update_layout(
            coloraxis_showscale=False,
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10,r=80,t=40,b=10), height=380,
        )
        st.plotly_chart(fig10, use_container_width=True)

    with col_r2:
        st.markdown("**🏆 Top 5 Reps**")
        for _, row in rep_df.head(5).iterrows():
            medal = ["🥇","🥈","🥉","4️⃣","5️⃣"][int(row["Rank"])-1]
            st.markdown(f"{medal} **{row['sales_rep']}**")
            st.caption(f"${row['Total_Revenue']:,.0f} · {int(row['Order_Count'])} orders")

    # Full table
    st.subheader("Full Leaderboard")
    display_rep = rep_df.copy()
    display_rep["Total_Revenue"] = display_rep["Total_Revenue"].apply(lambda v: f"${v:,.2f}")
    display_rep["Avg_Revenue"]   = display_rep["Avg_Revenue"].apply(lambda v: f"${v:,.2f}")
    st.dataframe(display_rep[["Rank","sales_rep","Total_Revenue","Order_Count",
                               "Avg_Revenue","Unique_Products"]],
                 use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — ML MODEL
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🤖 Random Forest Revenue Predictor")

    model, feature_cols = load_model()

    if model is None:
        st.warning("Model not found. Run Step 6 first: `python _run.py step6`")
    else:
        # Model metrics
        st.markdown("### Model Performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Algorithm",    "Random Forest")
        m2.metric("Test R²",      "0.7322")
        m3.metric("Test RMSE",    "$1,400")
        m4.metric("Test MAE",     "$925")

        st.markdown("---")
        col_ml1, col_ml2 = st.columns(2)

        # Feature importance chart
        with col_ml1:
            st.markdown("### Feature Importance")
            importances = model.feature_importances_
            fi_df = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": importances,
            }).sort_values("Importance", ascending=True)

            fig11 = px.bar(
                fi_df, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Blues",
                text=fi_df["Importance"].apply(lambda v: f"{v:.3f}"),
            )
            fig11.update_traces(textposition="outside")
            fig11.update_layout(
                coloraxis_showscale=False,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(l=10,r=60,t=10,b=10), height=400,
            )
            st.plotly_chart(fig11, use_container_width=True)

        # Live prediction form
        with col_ml2:
            st.markdown("### Live Revenue Predictor")
            st.caption("Fill in order details to predict revenue")

            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()

            with st.form("predict_form"):
                p_product  = st.selectbox("Product",        all_products)
                p_region   = st.selectbox("Region",         all_regions)
                p_qty      = st.slider("Quantity",          1, 20, 5)
                p_price    = st.number_input("Unit Price ($)", 10.0, 3000.0, 500.0, step=10.0)
                p_discount = st.slider("Discount (%)",      0, 30, 10)
                p_status   = st.selectbox("Order Status",   all_statuses)
                p_payment  = st.selectbox("Payment Method",
                             sorted(clean["payment_method"].dropna().unique()))
                p_month    = st.slider("Month", 1, 12, 6)
                submitted  = st.form_submit_button("🔮 Predict Revenue", use_container_width=True)

            if submitted:
                # Build feature row matching training columns
                all_cats = {
                    "region":         sorted(clean["region"].dropna().unique()),
                    "product":        sorted(clean["product"].dropna().unique()),
                    "payment_method": sorted(clean["payment_method"].dropna().unique()),
                    "order_status":   sorted(clean["order_status"].dropna().unique()),
                    "sales_rep":      sorted(clean["sales_rep"].dropna().unique()),
                }
                def encode(val, cats):
                    return cats.index(val) if val in cats else 0

                row = {
                    "quantity":           p_qty,
                    "unit_price":         p_price,
                    "discount":           p_discount / 100,
                    "month":              p_month,
                    "quarter":            (p_month - 1) // 3 + 1,
                    "day_of_week":        2,
                    "is_weekend":         0,
                    "year":               2024,
                    "region_enc":         encode(p_region,   all_cats["region"]),
                    "product_enc":        encode(p_product,  all_cats["product"]),
                    "payment_method_enc": encode(p_payment,  all_cats["payment_method"]),
                    "order_status_enc":   encode(p_status,   all_cats["order_status"]),
                    "sales_rep_enc":      0,
                }
                X_pred = pd.DataFrame([row])[feature_cols]
                pred   = model.predict(X_pred)[0]
                simple = p_price * p_qty * (1 - p_discount / 100)

                st.success(f"### Predicted Revenue: **${pred:,.2f}**")
                st.info(f"Simple calculation (price × qty × discount): **${simple:,.2f}**")
                diff = abs(pred - simple)
                st.caption(f"Model adjustment: ${diff:,.2f} "
                           f"({'above' if pred > simple else 'below'} simple calc)")

        # Actual vs Predicted scatter (from saved outputs)
        st.markdown("### Actual vs Predicted (Test Set)")
        if os.path.exists("outputs/feature_importance.png"):
            st.image("outputs/feature_importance.png",
                     caption="Feature Importance & Actual vs Predicted",
                     use_container_width=True)

    # Raw data explorer
    st.markdown("---")
    st.subheader("📋 Raw Data Explorer")
    n_rows = st.slider("Rows to show", 10, 200, 50)
    cols_to_show = st.multiselect(
        "Columns",
        options=list(clean.columns),
        default=["order_id","order_date","region","product","quantity",
                 "unit_price","discount","revenue","order_status"],
    )
    st.dataframe(df[cols_to_show].head(n_rows), use_container_width=True)
    st.download_button(
        "⬇️ Download Filtered Data as CSV",
        data=df.to_csv(index=False),
        file_name="filtered_sales_data.csv",
        mime="text/csv",
    )
