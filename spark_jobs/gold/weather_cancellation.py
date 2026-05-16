"""
gold/weather_cancellation.py
------------------------------
Gold layer — Weather Impact on Cancellations & Diversions (Full Year)

Sources (Silver):
    - cancelled_flights  (12 months — full 2023 — anchor)
    - weather            (12 months — full 2023 — full match possible)
    - airports           (static, 364 rows — broadcast)

THIS IS THE RICHEST GOLD TABLE.
Both cancelled and weather are full 12-month datasets.
Full-year picture of how weather conditions relate to cancellations.
Not available from flights (3 months only).

Join:
    cancelled LEFT JOIN weather ON
        cancelled.dep_airport = weather.iata_code
        AND cancelled.flightdate = weather.weather_date

    flightdate in cancelled is DateType (from Bronze).
    weather_date in weather is DateType. Types match — no cast needed.

Output:
    /home/asus/data_lake/gold/weather_cancellation/
    No partitioning — small aggregated summary

Grain: One row per wind_category + pressure_category + is_precipitation +
       weather_season + dep_region + flight_month

All column names lowercase to match Silver layer convention.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from configs.configs import SILVER_BASE, GOLD_BASE


SILVER_CANCELLED_PATH = f"{SILVER_BASE}/cancelled_flights/"
SILVER_WEATHER_PATH   = f"{SILVER_BASE}/weather/"
SILVER_AIRPORTS_PATH  = f"{SILVER_BASE}/airports/"
GOLD_PATH             = f"{GOLD_BASE}/weather_cancellation/"


def build_weather_cancellation(spark: SparkSession) -> None:

    print("=" * 60)
    print("Gold | weather_cancellation | Starting")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Read Silver tables
    # ------------------------------------------------------------------
    cancelled = spark.read.parquet(SILVER_CANCELLED_PATH)
    weather   = spark.read.parquet(SILVER_WEATHER_PATH)
    airports  = spark.read.parquet(SILVER_AIRPORTS_PATH)
    cancelled.cache()
    weather.cache()
    airports.cache()
    cancelled.count()
    weather.count()
    airports.count()
    print(f"[Silver] Cancelled rows      : {cancelled.count():,}")
    print(f"[Silver] Weather rows        : {weather.count():,}")
    print(f"[Silver] Airports rows       : {airports.count():,}")

    # ------------------------------------------------------------------
    # 2. Slim weather — only columns needed for this Gold job
    # ------------------------------------------------------------------
    weather_slim = weather.select(
        "iata_code",
        "weather_date",
        "weather_season",
        "wind_category",
        "pressure_category",
        "is_precipitation",
        "temp_avg_c",
        "temp_range_c",
        "wind_speed_kmh",
        "precipitation_mm",
        "snowfall_mm",
    )

    # ------------------------------------------------------------------
    # 3. LEFT JOIN cancelled → weather (cancelled is anchor)
    # ------------------------------------------------------------------
    joined = cancelled.join(
        weather_slim,
        (cancelled["dep_airport"] == weather_slim["iata_code"]) &
        (cancelled["flightdate"]  == weather_slim["weather_date"]),
        how="left"
    ).drop("iata_code", "weather_date")

    # ------------------------------------------------------------------
    # 4. Broadcast join departure airports
    # ------------------------------------------------------------------
    dep_airports = airports.select(
        F.col("iata_code"),
        F.col("us_region").alias("dep_region"),
        F.col("state").alias("dep_state"),
    )
    joined = joined.join(
        F.broadcast(dep_airports),
        joined["dep_airport"] == dep_airports["iata_code"],
        how="left"
    ).drop("iata_code")

    # ------------------------------------------------------------------
    # 5. Coverage check — all 12 months
    # ------------------------------------------------------------------
    total     = joined.count()
    matched   = joined.filter(F.col("wind_category").isNotNull()).count()
    match_pct = round(matched / total * 100, 1) if total > 0 else 0
    print(f"\n[Join] Total cancelled/diverted rows : {total:,}")
    print(f"[Join] Matched to weather            : {matched:,}  ({match_pct}%)")
    print(f"[Join] Unmatched                     : {total - matched:,}")

    print(f"[Join] Weather match coverage by month:")
    joined.groupBy("flight_month") \
          .agg(
              F.count("*").alias("total"),
              F.count(F.when(F.col("wind_category").isNotNull(), 1)).alias("matched"),
              F.round(
                  F.count(F.when(F.col("wind_category").isNotNull(), 1)) /
                  F.count("*") * 100, 1
              ).alias("match_pct")
          ).orderBy("flight_month").show()

    # ------------------------------------------------------------------
    # 6. Aggregate
    # ------------------------------------------------------------------
    cancelled_flag = (F.col("cancelled") == 1).cast("int")
    diverted_flag  = (F.col("diverted")  == 1).cast("int")
    has_weather    = F.col("wind_category").isNotNull().cast("int")

    gold_df = joined.groupBy(
        "weather_season",
        "wind_category",
        "pressure_category",
        "is_precipitation",
        "dep_region",
        "flight_month",
    ).agg(
        F.count("*")                                       .alias("total_events"),
        F.sum(has_weather)                                 .alias("events_with_weather"),
        F.round(F.avg(has_weather) * 100, 1)               .alias("weather_match_pct"),
        F.sum(cancelled_flag)                              .alias("cancelled_count"),
        F.sum(diverted_flag)                               .alias("diverted_count"),
        F.round(F.avg(cancelled_flag) * 100, 2)            .alias("cancellation_rate_pct"),
        F.round(F.avg(diverted_flag)  * 100, 2)            .alias("diversion_rate_pct"),
        F.round(F.avg("temp_avg_c"), 1)                    .alias("avg_temp_c"),
        F.round(F.avg("temp_range_c"), 1)                  .alias("avg_temp_range_c"),
        F.round(F.avg("wind_speed_kmh"), 1)                .alias("avg_wind_speed_kmh"),
        F.round(F.avg("precipitation_mm"), 2)              .alias("avg_precipitation_mm"),
        F.round(F.avg("snowfall_mm"), 2)                   .alias("avg_snowfall_mm"),
    )

    # ------------------------------------------------------------------
    # 7. Sanity checks — key business insights
    # ------------------------------------------------------------------
    gold_count = gold_df.count()
    print(f"\n[Gold] Output rows           : {gold_count:,}")
    print(f"[Gold] Output columns        : {len(gold_df.columns)}")

    print(f"\n[Gold] Cancellation rate by wind_category (full year):")
    gold_df.groupBy("wind_category") \
           .agg(
               F.sum("total_events").alias("total_events"),
               F.sum("cancelled_count").alias("cancelled"),
               F.round(F.avg("cancellation_rate_pct"), 2).alias("cancel_rate_pct"),
               F.round(F.avg("avg_wind_speed_kmh"), 1).alias("avg_wind_kmh"),
           ).orderBy(F.col("cancel_rate_pct").desc()).show(truncate=False)

    print(f"[Gold] Cancellation rate by pressure_category:")
    gold_df.groupBy("pressure_category") \
           .agg(
               F.sum("total_events").alias("total_events"),
               F.round(F.avg("cancellation_rate_pct"), 2).alias("cancel_rate_pct"),
           ).orderBy(F.col("cancel_rate_pct").desc()).show(truncate=False)

    print(f"[Gold] Precipitation vs no precipitation:")
    gold_df.groupBy("is_precipitation") \
           .agg(
               F.sum("total_events").alias("total_events"),
               F.round(F.avg("cancellation_rate_pct"), 2).alias("cancel_rate_pct"),
               F.round(F.avg("avg_precipitation_mm"), 2).alias("avg_precip_mm"),
           ).orderBy("is_precipitation").show(truncate=False)

    print(f"[Gold] Cancellation rate by season (full year):")
    gold_df.groupBy("weather_season") \
           .agg(
               F.sum("total_events").alias("total_events"),
               F.sum("cancelled_count").alias("cancelled"),
               F.round(F.avg("cancellation_rate_pct"), 2).alias("cancel_rate_pct"),
               F.round(F.avg("avg_snowfall_mm"), 2).alias("avg_snowfall_mm"),
           ).orderBy(F.col("cancel_rate_pct").desc()).show(truncate=False)

    print(f"[Gold] Cancellation rate by region:")
    gold_df.groupBy("dep_region") \
           .agg(
               F.sum("total_events").alias("total_events"),
               F.round(F.avg("cancellation_rate_pct"), 2).alias("cancel_rate_pct"),
           ).orderBy(F.col("cancel_rate_pct").desc()).show(truncate=False)

    # ------------------------------------------------------------------
    # 8. Write Gold parquet
    # ------------------------------------------------------------------
    gold_df.write \
           .mode("overwrite") \
           .parquet(GOLD_PATH)

    print(f"\n[Gold] Written to            : {GOLD_PATH}")
    cancelled.unpersist()
    weather.unpersist()
    airports.unpersist()
    print("=" * 60)
    print("Gold | weather_cancellation | Complete ✓")
    print("=" * 60)


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("Gold_WeatherCancellation") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    build_weather_cancellation(spark)
    spark.stop()