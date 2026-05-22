"""
Step 4 — PySpark Pipeline with Window Functions & Parquet Output
Demonstrates distributed data processing skills that separate you from
other candidates:
  • SparkSession setup
  • Schema definition
  • DataFrame transformations
  • Window functions (rank, lag, running total, moving average)
  • Parquet read/write
Output: data/spark_output/ (Parquet)
"""

import os
import warnings

warnings.filterwarnings("ignore")

# ── PySpark imports ───────────────────────────────────────────────────────────
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

CLEAN_CSV = "data/cleaned_sales_data.csv"
PARQUET_OUT = "data/spark_output"


# ─────────────────────────────────────────────────────────────────────────────
# 1. SPARK SESSION
# ─────────────────────────────────────────────────────────────────────────────

def create_spark_session() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("SalesDataPipeline")
        .master("local[*]")                          # use all local CPU cores
        .config("spark.sql.shuffle.partitions", "4") # small dataset → fewer partitions
        .config("spark.driver.memory", "2g")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")          # suppress INFO/WARN noise
    print(f"✅ Spark version : {spark.version}")
    print(f"   Master        : {spark.sparkContext.master}")
    return spark


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data(spark: SparkSession, path: str):
    """Load cleaned CSV with an explicit schema for type safety."""
    schema = StructType([
        StructField("order_id",        StringType(),  True),
        StructField("order_date",      StringType(),  True),   # parse later
        StructField("region",          StringType(),  True),
        StructField("product",         StringType(),  True),
        StructField("category",        StringType(),  True),
        StructField("sales_rep",       StringType(),  True),
        StructField("quantity",        IntegerType(), True),
        StructField("unit_price",      DoubleType(),  True),
        StructField("discount",        DoubleType(),  True),
        StructField("revenue",         DoubleType(),  True),
        StructField("payment_method",  StringType(),  True),
        StructField("order_status",    StringType(),  True),
        StructField("customer_id",     StringType(),  True),
    ])

    df = (
        spark.read
        .option("header", "true")
        .option("nullValue", "")
        .schema(schema)
        .csv(path)
    )

    # Parse date string → DateType
    df = df.withColumn("order_date", F.to_date("order_date", "yyyy-MM-dd"))

    print(f"\n✅ Loaded {df.count():,} rows, {len(df.columns)} columns")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRANSFORMATIONS
# ─────────────────────────────────────────────────────────────────────────────

def transform_data(df):
    """Add derived columns using Spark SQL functions."""
    print("\n── Transformations ──────────────────────────────────────")

    df = (
        df
        # Date features
        .withColumn("year",         F.year("order_date"))
        .withColumn("month",        F.month("order_date"))
        .withColumn("quarter",      F.quarter("order_date"))
        .withColumn("month_year",   F.date_format("order_date", "yyyy-MM"))
        .withColumn("day_of_week",  F.date_format("order_date", "EEEE"))
        .withColumn("is_weekend",   F.dayofweek("order_date").isin([1, 7]))

        # Revenue features
        .withColumn("gross_revenue",    F.round(F.col("unit_price") * F.col("quantity"), 2))
        .withColumn("discount_amount",  F.round(F.col("gross_revenue") * F.col("discount"), 2))
        .withColumn("revenue_per_unit", F.round(F.col("revenue") / F.col("quantity"), 2))

        # Discount tier
        .withColumn(
            "discount_tier",
            F.when(F.col("discount") < 0.05, "None (<5%)")
             .when(F.col("discount") < 0.15, "Low (5-15%)")
             .when(F.col("discount") < 0.25, "Medium (15-25%)")
             .otherwise("High (>25%)")
        )

        # Order status flag
        .withColumn("is_completed", F.col("order_status") == "Completed")
    )

    print(f"   Columns after transform: {len(df.columns)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. WINDOW FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def apply_window_functions(df):
    """
    Window functions are a key differentiator in interviews.
    Demonstrates: rank, dense_rank, lag, lead, running total, moving avg.
    """
    print("\n── Window Functions ─────────────────────────────────────")

    # ── Window specs ──────────────────────────────────────────────────────────

    # Rank sales reps by revenue within each region
    region_window = (
        Window.partitionBy("region")
              .orderBy(F.desc("revenue"))
    )

    # Running total of revenue per region, ordered by date
    region_date_window = (
        Window.partitionBy("region")
              .orderBy("order_date")
              .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    # Lag/lead within each product, ordered by date
    product_date_window = (
        Window.partitionBy("product")
              .orderBy("order_date")
    )

    # Moving 7-day average revenue per region
    moving_avg_window = (
        Window.partitionBy("region")
              .orderBy(F.col("order_date").cast("long"))
              .rangeBetween(-6 * 86400, 0)   # 6 days back in seconds
    )

    # Monthly rank of products by revenue
    monthly_product_window = (
        Window.partitionBy("month_year", "product")
              .orderBy(F.desc("revenue"))
    )

    df = (
        df
        # Rank within region
        .withColumn("rank_in_region",       F.rank().over(region_window))
        .withColumn("dense_rank_in_region", F.dense_rank().over(region_window))

        # Running total revenue per region
        .withColumn("running_revenue_region", F.round(F.sum("revenue").over(region_date_window), 2))

        # Lag: previous order revenue for same product
        .withColumn("prev_order_revenue",   F.lag("revenue", 1).over(product_date_window))

        # Lead: next order revenue for same product
        .withColumn("next_order_revenue",   F.lead("revenue", 1).over(product_date_window))

        # Revenue change vs previous order (same product)
        .withColumn(
            "revenue_change",
            F.round(F.col("revenue") - F.col("prev_order_revenue"), 2)
        )

        # 7-day moving average revenue per region
        .withColumn("moving_avg_7d", F.round(F.avg("revenue").over(moving_avg_window), 2))

        # Percent of region total revenue
        .withColumn(
            "pct_of_region_revenue",
            F.round(
                F.col("revenue") / F.sum("revenue").over(Window.partitionBy("region")) * 100,
                2
            )
        )

        # Monthly product rank
        .withColumn("monthly_product_rank", F.rank().over(monthly_product_window))
    )

    print("   Window columns added:")
    print("     rank_in_region, dense_rank_in_region")
    print("     running_revenue_region")
    print("     prev_order_revenue, next_order_revenue, revenue_change")
    print("     moving_avg_7d, pct_of_region_revenue")
    print("     monthly_product_rank")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. AGGREGATIONS (Spark SQL style)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_data(df):
    """Run aggregations using both DataFrame API and Spark SQL."""
    print("\n── Aggregations ─────────────────────────────────────────")

    # Register as temp view for SQL queries
    df.createOrReplaceTempView("sales")

    spark = df.sparkSession

    # Region summary via SQL
    region_sql = spark.sql("""
        SELECT
            region,
            COUNT(*)                        AS total_orders,
            ROUND(SUM(revenue), 2)          AS total_revenue,
            ROUND(AVG(revenue), 2)          AS avg_revenue,
            ROUND(AVG(discount) * 100, 2)   AS avg_discount_pct,
            COUNT(DISTINCT customer_id)     AS unique_customers,
            COUNT(DISTINCT product)         AS unique_products
        FROM sales
        GROUP BY region
        ORDER BY total_revenue DESC
    """)
    print(f"   Region summary rows  : {region_sql.count()}")

    # Monthly trend via DataFrame API
    monthly_df = (
        df.groupBy("month_year")
          .agg(
              F.count("order_id").alias("total_orders"),
              F.round(F.sum("revenue"), 2).alias("total_revenue"),
              F.round(F.avg("revenue"), 2).alias("avg_order_value"),
              F.round(F.sum("quantity"), 0).alias("total_units"),
          )
          .orderBy("month_year")
    )
    print(f"   Monthly trend rows   : {monthly_df.count()}")

    # Product performance
    product_df = (
        df.groupBy("product")
          .agg(
              F.round(F.sum("revenue"), 2).alias("total_revenue"),
              F.sum("quantity").alias("total_units"),
              F.count("order_id").alias("order_count"),
              F.round(F.avg("unit_price"), 2).alias("avg_price"),
          )
          .orderBy(F.desc("total_revenue"))
    )
    print(f"   Product summary rows : {product_df.count()}")

    return {
        "region": region_sql,
        "monthly": monthly_df,
        "product": product_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. WRITE PARQUET
# ─────────────────────────────────────────────────────────────────────────────

def write_parquet(df, aggs: dict, base_path: str) -> None:
    """Write main DataFrame and aggregations as Parquet (columnar format)."""
    print("\n── Writing Parquet ───────────────────────────────────────")
    os.makedirs(base_path, exist_ok=True)

    # Main enriched dataset — partitioned by region for query efficiency
    main_path = f"{base_path}/enriched_sales"
    (
        df.repartition(1)          # single file for small dataset
          .write
          .mode("overwrite")
          .partitionBy("region")   # partition pruning in downstream queries
          .parquet(main_path)
    )
    print(f"   ✅ Enriched sales → {main_path}  (partitioned by region)")

    # Aggregation outputs
    for name, agg_df in aggs.items():
        path = f"{base_path}/{name}_agg"
        agg_df.coalesce(1).write.mode("overwrite").parquet(path)
        print(f"   ✅ {name:<10} agg  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Sales Data Pipeline — Step 4: PySpark Pipeline")
    print("=" * 60)

    spark = create_spark_session()

    df = load_data(spark, CLEAN_CSV)
    df = transform_data(df)
    df = apply_window_functions(df)
    aggs = aggregate_data(df)
    write_parquet(df, aggs, PARQUET_OUT)

    # Show sample output
    print("\n📋 Sample enriched rows (key columns):")
    df.select(
        "order_id", "region", "product", "revenue",
        "rank_in_region", "running_revenue_region",
        "moving_avg_7d", "pct_of_region_revenue"
    ).show(5, truncate=False)

    print("\n📊 Region summary:")
    aggs["region"].show(truncate=False)

    spark.stop()
    print("\n✅ Step 4 complete! Spark session stopped.")
