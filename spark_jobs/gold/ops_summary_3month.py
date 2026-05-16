"""
gold/ops_summary_3month.py
---------------------------
Gold layer — Real Operational Summary (Jan–Mar 2023)

The Problem with standalone cancellation_analysis.py:
    The cancelled Silver table only contains cancelled and diverted flights.
    Any rate computed within that table uses a biased denominator — it is
    already a 100% failure dataset. Cancellation rates of 80-92% are
    meaningless because the denominator excludes all completed flights.

The Fix — Union approach:
    Total Scheduled = Completed flights (from silver/flights)
                    + Cancelled flights (from silver/cancelled_flights, 3 months)
                    + Diverted  flights (from silver/cancelled_flights, 3 months)

    Real Cancellation Rate = Cancelled / Total Scheduled
    Real Completion Rate   = Completed / Total Scheduled
    Real Diversion Rate    = Diverted  / Total Scheduled

Sources (Silver):
    - flights            (3 months Jan/Feb/Mar 2023 — completed flights only)
    - cancelled_flights  (12 months — filtered to Jan/Feb/Mar for overlap)
    - airports           (static, 364 rows — broadcast)

Output:
    /home/asus/data_lake/gold/ops_summary_3month/
    Partitioned by: flight_year, flight_month

Grain: One row per airline + dep_airport + arr_airport +
       flight_month + distance_type + weekday_name

Metrics (all computed against real total_scheduled denominator):
    - total_scheduled        (completed + cancelled + diverted)
    - completed_count
    - cancelled_count
    - diverted_count
    - completion_rate_pct    (completed / total_scheduled * 100)
    - cancellation_rate_pct  (cancelled / total_scheduled * 100)
    - diversion_rate_pct     (diverted  / total_scheduled * 100)
    - avg_dep_delay_min      (completed + diverted flights — has real dep_delay)
    - avg_arr_delay_min      (completed flights only — cancelled/diverted have no arrival)
    - avg_flight_duration_min
    - on_time_pct            (dep_delay <= 0 among completed + diverted)
    - delayed_pct            (dep_delay > 0 among completed + diverted)
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from configs.configs import SILVER_BASE, GOLD_BASE


SILVER_FLIGHTS_PATH   = f"{SILVER_BASE}/flights/"
SILVER_CANCELLED_PATH = f"{SILVER_BASE}/cancelled_flights/"
SILVER_AIRPORTS_PATH  = f"{SILVER_BASE}/airports/"
GOLD_PATH             = f"{GOLD_BASE}/ops_summary_3month/"

# Months present in the flights dataset — filter cancelled to match
FLIGHT_MONTHS = [1, 2, 3]


def build_ops_summary(spark: SparkSession) -> None:

    print("=" * 60)
    print("Gold | ops_summary_3month | Starting")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Read Silver tables
    # ------------------------------------------------------------------
    flights   = spark.read.parquet(SILVER_FLIGHTS_PATH)
    cancelled = spark.read.parquet(SILVER_CANCELLED_PATH)
    airports  = spark.read.parquet(SILVER_AIRPORTS_PATH)
    flights.cache()
    cancelled.cache()
    airports.cache()
    flights.count()
    cancelled.count()
    airports.count()
    flights_count   = flights.count()
    cancelled_count = cancelled.count()
    print(f"[Silver] Flights rows (completed) : {flights_count:,}")
    print(f"[Silver] Cancelled rows (all 12m) : {cancelled_count:,}")

    # ------------------------------------------------------------------
    # 2. Filter cancelled to the 3-month overlap window only
    # ------------------------------------------------------------------
    cancelled_3m = cancelled.filter(F.col("flight_month").isin(FLIGHT_MONTHS))
    c3_count = cancelled_3m.count()
    print(f"[Filter] Cancelled rows (Jan-Mar) : {c3_count:,}")
    print(f"[Scope] Total scheduled (union)   : {flights_count + c3_count:,}")

    # ------------------------------------------------------------------
    # 3. Build a common schema for union
    #    Only columns present in BOTH datasets are included.
    #    dep_delay / arr_delay are included — null for Cancelled rows
    #    (already nullified in Silver), real values for Completed/Diverted.
    # ------------------------------------------------------------------

    COMMON_COLS = [
        "flightdate", "flight_year", "flight_month", "flight_day",
        "day_of_week", "weekday_name",
        "airline", "tail_number",
        "dep_airport", "dep_cityname",
        "arr_airport", "arr_cityname",
        "deptime_label", "distance_type",
        "dep_delay", "arr_delay", "flight_duration",
    ]

    # Flights Silver has no flight_status — tag all rows as Completed
    flights_slim = flights.select(*COMMON_COLS) \
                          .withColumn("flight_status", F.lit("Completed"))

    # Cancelled Silver already has flight_status (Cancelled / Diverted)
    cancelled_slim = cancelled_3m.select(*COMMON_COLS, "flight_status")

    # ------------------------------------------------------------------
    # 4. Union into one operational view
    # ------------------------------------------------------------------
    all_flights = flights_slim.union(cancelled_slim)
    total = all_flights.count()
    print(f"[Union] Total rows after union    : {total:,}")

    # Quick status split — sanity check
    print(f"[Union] Status breakdown:")
    all_flights.groupBy("flight_status").count() \
               .orderBy("flight_status").show()

    # ------------------------------------------------------------------
    # 5. Broadcast join departure airports for regional enrichment
    # ------------------------------------------------------------------
    dep_airports = airports.select(
        F.col("iata_code"),
        F.col("city").alias("dep_city"),
        F.col("state").alias("dep_state"),
        F.col("us_region").alias("dep_region"),
    )
    all_flights = all_flights.join(
        F.broadcast(dep_airports),
        all_flights["dep_airport"] == dep_airports["iata_code"],
        how="left"
    ).drop("iata_code")

    # ------------------------------------------------------------------
    # 6. Aggregate against the real total_scheduled denominator
    # ------------------------------------------------------------------
    completed_flag = (F.col("flight_status") == "Completed").cast("int")
    cancelled_flag = (F.col("flight_status") == "Cancelled").cast("int")
    diverted_flag  = (F.col("flight_status") == "Diverted").cast("int")

    # Delay flags — only meaningful when the flight actually departed
    # (Completed + Diverted have real dep_delay; Cancelled have null)
    delayed_flag  = (F.col("dep_delay") > 0).cast("int")
    on_time_flag  = (F.col("dep_delay") <= 0).cast("int")

    gold_df = all_flights.groupBy(
        "flight_year", "flight_month",
        "airline",
        "dep_airport", "dep_city", "dep_state", "dep_region",
        "arr_airport", "arr_cityname",
        "distance_type",
        "weekday_name",
    ).agg(
        # Volume
        F.count("*")                                             .alias("total_scheduled"),
        F.sum(completed_flag)                                    .alias("completed_count"),
        F.sum(cancelled_flag)                                    .alias("cancelled_count"),
        F.sum(diverted_flag)                                     .alias("diverted_count"),

        # Real rates — denominator is total_scheduled
        F.round(F.avg(completed_flag) * 100, 2)                  .alias("completion_rate_pct"),
        F.round(F.avg(cancelled_flag) * 100, 2)                  .alias("cancellation_rate_pct"),
        F.round(F.avg(diverted_flag)  * 100, 2)                  .alias("diversion_rate_pct"),

        # Delay metrics — avg() naturally ignores nulls (cancelled rows)
        # so these averages are only over flights that actually departed
        F.round(F.avg("dep_delay"), 1)                           .alias("avg_dep_delay_min"),
        F.round(F.avg("arr_delay"), 1)                           .alias("avg_arr_delay_min"),
        F.round(F.avg("flight_duration"), 1)                     .alias("avg_flight_duration_min"),

        # On-time / delayed — among departed flights (null dep_delay excluded by avg)
        F.round(
            F.sum(F.when(F.col("dep_delay").isNotNull(), on_time_flag)) /
            F.sum(F.when(F.col("dep_delay").isNotNull(), F.lit(1))) * 100, 1
        )                                                        .alias("on_time_pct"),
        F.round(
            F.sum(F.when(F.col("dep_delay").isNotNull(), delayed_flag)) /
            F.sum(F.when(F.col("dep_delay").isNotNull(), F.lit(1))) * 100, 1
        )                                                        .alias("delayed_pct"),
    )

    # ------------------------------------------------------------------
    # 7. Sanity checks — now the numbers should make sense
    # ------------------------------------------------------------------
    gold_count = gold_df.count()
    print(f"\n[Gold] Output rows               : {gold_count:,}")
    print(f"[Gold] Output columns            : {len(gold_df.columns)}")

    print(f"\n[Gold] REAL cancellation rate by airline (Jan-Mar 2023):")
    gold_df.groupBy("airline") \
           .agg(
               F.sum("total_scheduled").alias("total_scheduled"),
               F.sum("completed_count").alias("completed"),
               F.sum("cancelled_count").alias("cancelled"),
               F.round(F.avg("cancellation_rate_pct"), 2).alias("cancel_rate_pct"),
               F.round(F.avg("completion_rate_pct"), 2).alias("completion_rate_pct"),
           ).orderBy(F.col("cancel_rate_pct").asc()).show(15, truncate=False)

    print(f"[Gold] REAL cancellation rate by month:")
    gold_df.groupBy("flight_month") \
           .agg(
               F.sum("total_scheduled").alias("total_scheduled"),
               F.sum("cancelled_count").alias("cancelled"),
               F.round(F.avg("cancellation_rate_pct"), 2).alias("cancel_rate_pct"),
               F.round(F.avg("completion_rate_pct"), 2).alias("completion_rate_pct"),
           ).orderBy("flight_month").show()

    print(f"[Gold] REAL cancellation rate by region:")
    gold_df.groupBy("dep_region") \
           .agg(
               F.sum("total_scheduled").alias("total_scheduled"),
               F.sum("cancelled_count").alias("cancelled"),
               F.round(F.avg("cancellation_rate_pct"), 2).alias("cancel_rate_pct"),
               F.round(F.avg("on_time_pct"), 1).alias("on_time_pct"),
           ).orderBy(F.col("cancel_rate_pct").desc()).show(truncate=False)

    print(f"[Gold] REAL performance by weekday (cancellation + on-time among departed):")
    gold_df.groupBy("weekday_name") \
           .agg(
               F.sum("total_scheduled").alias("total_scheduled"),
               F.round(F.avg("cancellation_rate_pct"), 2).alias("cancel_rate_pct"),
               F.round(F.avg("on_time_pct"), 1).alias("on_time_pct"),
               F.round(F.avg("avg_dep_delay_min"), 1).alias("avg_dep_delay_min"),
           ).orderBy(F.col("cancel_rate_pct").desc()).show(truncate=False)

    print(f"[Gold] Top 10 worst routes by cancellation rate (min 50 scheduled):")
    gold_df.filter(F.col("total_scheduled") >= 50) \
           .orderBy(F.col("cancellation_rate_pct").desc()) \
           .select(
               "dep_airport", "arr_airport", "airline",
               "total_scheduled", "cancelled_count",
               "cancellation_rate_pct", "avg_dep_delay_min"
           ).show(10, truncate=False)

    # ------------------------------------------------------------------
    # 8. Write Gold parquet
    # ------------------------------------------------------------------
    gold_df.write \
           .mode("overwrite") \
           .partitionBy("flight_year", "flight_month") \
           .parquet(GOLD_PATH)

    print(f"\n[Gold] Written to                : {GOLD_PATH}")
    flights.unpersist()
    cancelled.unpersist()
    airports.unpersist()
    print(f"[Gold] Partitioned by            : flight_year, flight_month")
    print("=" * 60)
    print("Gold | ops_summary_3month | Complete ✓")
    print("=" * 60)


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("Gold_OpsSummary3Month") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    build_ops_summary(spark)
    spark.stop()