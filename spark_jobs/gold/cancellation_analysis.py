"""
gold/cancellation_analysis.py
------------------------------
Gold layer — Cancellation & Diversion Analysis (Full Year)

Sources (Silver):
    - cancelled_flights  (12 months — full 2023)
    - airports           (static, 364 rows — broadcast)

Output:
    /home/asus/data_lake/gold/cancellation_analysis/
    Partitioned by: flight_year, flight_month

Grain: One row per airline + dep_airport + arr_airport +
       flight_month + distance_type + weekday_name

12-month scope. Do NOT compare raw counts against the
3-month flight_performance Gold table.

All column names lowercase to match Silver layer convention.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from configs.configs import SILVER_BASE, GOLD_BASE


SILVER_CANCELLED_PATH = f"{SILVER_BASE}/cancelled_flights/"
SILVER_AIRPORTS_PATH  = f"{SILVER_BASE}/airports/"
GOLD_PATH             = f"{GOLD_BASE}/cancellation_analysis/"


def build_cancellation_analysis(spark: SparkSession) -> None:

    print("=" * 60)
    print("Gold | cancellation_analysis | Starting")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Read Silver tables
    # ------------------------------------------------------------------
    cancelled = spark.read.parquet(SILVER_CANCELLED_PATH)
    airports  = spark.read.parquet(SILVER_AIRPORTS_PATH)

    cancelled.cache()
    airports.cache()
    cancelled.count()
    airports.count()
    airports

    print(f"[Silver] Cancelled rows      : {cancelled.count():,}")
    print(f"[Silver] Airports rows       : {airports.count():,}")

    # ------------------------------------------------------------------
    # 2. Broadcast join departure airports
    # ------------------------------------------------------------------
    dep_airports = airports.select(
        F.col("iata_code"),
        F.col("city").alias("dep_city"),
        F.col("state").alias("dep_state"),
        F.col("us_region").alias("dep_region"),
    )

    cancelled = cancelled.join(
        F.broadcast(dep_airports),
        cancelled["dep_airport"] == dep_airports["iata_code"],
        how="left"
    ).drop("iata_code")

    # ------------------------------------------------------------------
    # 3. Aggregate
    # ------------------------------------------------------------------
    cancelled_flag = (F.col("cancelled") == 1).cast("int")
    diverted_flag  = (F.col("diverted")  == 1).cast("int")

    gold_df = cancelled.groupBy(
        "flight_year", "flight_month",
        "airline",
        "dep_airport", "dep_city", "dep_state", "dep_region",
        "arr_airport", "arr_cityname",
        "distance_type",
        "weekday_name",
    ).agg(
        F.count("*")                                         .alias("total_events"),
        F.sum(cancelled_flag)                                .alias("cancelled_count"),
        F.sum(diverted_flag)                                 .alias("diverted_count"),
        F.round(F.avg(cancelled_flag) * 100, 2)              .alias("cancellation_rate_pct"),
        F.round(F.avg(diverted_flag)  * 100, 2)              .alias("diversion_rate_pct"),
        # Delay metrics for diverted rows only — cancelled rows have null delay
        F.round(
            F.avg(F.when(F.col("diverted") == 1, F.col("dep_delay"))), 1
        )                                                    .alias("diverted_avg_dep_delay_min"),
        F.max(
            F.when(F.col("diverted") == 1, F.col("dep_delay"))
        )                                                    .alias("diverted_max_dep_delay_min"),
    )

    # ------------------------------------------------------------------
    # 4. Sanity checks
    # ------------------------------------------------------------------
    gold_count = gold_df.count()
    print(f"\n[Gold] Output rows           : {gold_count:,}")
    print(f"[Gold] Output columns        : {len(gold_df.columns)}")

    print(f"\n[Gold] Cancellation rate by airline:")
    gold_df.groupBy("airline") \
           .agg(
               F.sum("total_events").alias("total_events"),
               F.sum("cancelled_count").alias("cancelled"),
               F.round(F.avg("cancellation_rate_pct"), 2).alias("avg_cancel_rate_pct"),
           ).orderBy(F.col("avg_cancel_rate_pct").desc()).show(15, truncate=False)

    print(f"[Gold] Cancellation rate by month:")
    gold_df.groupBy("flight_month") \
           .agg(
               F.sum("total_events").alias("total_events"),
               F.sum("cancelled_count").alias("cancelled"),
               F.round(F.avg("cancellation_rate_pct"), 2).alias("cancel_rate_pct"),
           ).orderBy("flight_month").show(12, truncate=False)

    print(f"[Gold] Cancellation rate by region:")
    gold_df.groupBy("dep_region") \
           .agg(
               F.sum("total_events").alias("total_events"),
               F.round(F.avg("cancellation_rate_pct"), 2).alias("cancel_rate_pct"),
           ).orderBy(F.col("cancel_rate_pct").desc()).show(truncate=False)

    print(f"[Gold] Cancellation rate by weekday:")
    gold_df.groupBy("weekday_name") \
           .agg(
               F.round(F.avg("cancellation_rate_pct"), 2).alias("cancel_rate_pct"),
               F.sum("total_events").alias("total_events"),
           ).orderBy(F.col("cancel_rate_pct").desc()).show(truncate=False)

    # ------------------------------------------------------------------
    # 5. Write Gold parquet
    # ------------------------------------------------------------------
    gold_df.write \
           .mode("overwrite") \
           .partitionBy("flight_year", "flight_month") \
           .parquet(GOLD_PATH)

    print(f"\n[Gold] Written to            : {GOLD_PATH}")
    cancelled.unpersist()
    airports.unpersist()
    print(f"[Gold] Partitioned by        : flight_year, flight_month")
    print("=" * 60)
    print("Gold | cancellation_analysis | Complete ✓")
    print("=" * 60)


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("Gold_CancellationAnalysis") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    build_cancellation_analysis(spark)

    spark.stop()