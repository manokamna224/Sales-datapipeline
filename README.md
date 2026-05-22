# Sales Data Pipeline — Data Engineering Portfolio Project
# in this file there a video files that show my project how its work 
A complete, interview-ready data engineering project built across 6 progressive steps.

---

## Project Structure

```
sales_pipeline/
├── step1_setup.py            # Install all dependencies
├── step2_generate_data.py    # Generate 1000 rows of messy sales data
├── step3_pandas_pipeline.py  # Clean, engineer features, aggregate (Pandas + NumPy)
├── step4_pyspark_pipeline.py # Distributed pipeline with window functions (PySpark)
├── step5_dashboard.py        # 4-chart analytics dashboard (Matplotlib + Seaborn)
├── step6_ml_model.py         # Random Forest revenue prediction (scikit-learn)
├── run_pipeline.py           # Run all steps in sequence
├── requirements.txt          # Pinned dependencies
│
├── data/                     # Generated at runtime
│   ├── raw_sales_data.csv
│   ├── cleaned_sales_data.csv
│   ├── aggregations/
│   └── spark_output/         # Parquet files
│
├── outputs/                  # Generated at runtime
│   ├── sales_dashboard.png
│   └── feature_importance.png
│
└── models/                   # Generated at runtime
    └── revenue_model.joblib
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
# or run the setup script:
python step1_setup.py
```

### 2. Run all steps
```bash
cd sales_pipeline
python run_pipeline.py
```

### 3. Run individual steps
```bash
python step2_generate_data.py
python step3_pandas_pipeline.py
python step4_pyspark_pipeline.py   # requires Java 8+
python step5_dashboard.py
python step6_ml_model.py
```

### 4. Run specific steps (skip PySpark if Java not available)
```bash
python run_pipeline.py --steps 2 3 5 6
python run_pipeline.py --skip 4
```

---

## What Each Step Covers

### Step 1 — Environment Setup
- Installs all required packages via pip
- Verifies imports after installation

### Step 2 — Data Generation
- 1000 rows of realistic sales data using Faker
- Intentional messiness: 5% nulls, 30 duplicate rows, negative quantities, future dates, mixed-case strings
- Columns: order_id, order_date, region, product, sales_rep, quantity, unit_price, discount, revenue, payment_method, order_status, customer_id

### Step 3 — Pandas Pipeline *(core — 70% of interview topics)*
**Cleaning:**
- Deduplication (`drop_duplicates`)
- String normalisation (strip, title case)
- Date parsing + future date handling
- Negative value correction
- Null imputation: median for numerics, mode for categoricals, forward-fill for dates
- Outlier removal via IQR method

**Feature Engineering:**
- Date features: year, month, quarter, day_of_week, is_weekend, month_year
- Revenue features: revenue_per_unit, discount_amount, gross_revenue
- Binning: discount_tier (pd.cut), revenue_tier (pd.qcut)
- Flags: is_high_value, days_since_order

**Aggregations:**
- GroupBy: region, monthly, product, sales rep
- Rolling 3-month average
- Pivot table (region × product)
- Revenue share percentage

### Step 4 — PySpark Pipeline *(differentiator)*
- SparkSession with local mode
- Explicit schema definition
- DataFrame API + Spark SQL
- **Window functions:**
  - `rank()`, `dense_rank()` within region
  - Running total (`sum` over unbounded window)
  - `lag()` / `lead()` for previous/next order revenue
  - 7-day moving average (`rangeBetween`)
  - Percent of region total
- Parquet output partitioned by region

### Step 5 — Dashboard
4-chart layout (18×12 inches):
1. **Revenue by Region** — horizontal bar with value labels
2. **Monthly Revenue Trend** — bar + 3-month rolling average line
3. **Product Revenue Share** — donut chart with top 6 products
4. **Order Value Distribution** — histogram + KDE + quartile lines

### Step 6 — ML Model
- Feature encoding (LabelEncoder for categoricals)
- 80/20 train-test split
- 5-fold cross-validation (R², RMSE)
- GridSearchCV hyperparameter tuning
- Evaluation: RMSE, MAE, R², MAPE
- Feature importance bar chart + Actual vs Predicted scatter
- Model persistence with joblib

---

## Interview Talking Points

| Topic | Where it appears |
|---|---|
| Null handling strategies | Step 3 — median/mode/ffill |
| Deduplication | Step 3 — drop_duplicates |
| Outlier detection | Step 3 — IQR method |
| Feature engineering | Steps 3, 6 |
| GroupBy + aggregations | Steps 3, 4 |
| Pivot tables | Step 3 |
| Rolling windows | Steps 3, 4 |
| PySpark window functions | Step 4 |
| Parquet / columnar storage | Step 4 |
| Data visualisation | Step 5 |
| ML pipeline | Step 6 |
| Cross-validation | Step 6 |
| Hyperparameter tuning | Step 6 |
| Model persistence | Step 6 |

---

## Prerequisites
- Python 3.10+
- Java 8+ (for PySpark — Step 4 only)
