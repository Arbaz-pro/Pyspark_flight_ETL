"""
gold/weather_delay_impact.py
------------------------------
Gold layer — Weather Impact on Flight Delays

Sources (Silver):
    - flights   (3 months — Jan/Feb/Mar 2023 — anchor)
    - weather   (12 months — only 3 months will match flight dates)
    - airports  (static, 364 rows — broadcast)

Join:
    flights LEFT JOIN weather ON
        flights.dep_airport = weather.iata_code
        AND flights.flightdate = weather.weather_date

Output:
    /home/asus/data_lake/gold/weather_delay_impact/
    No partitioning — small aggregated summary

Grain: One row per wind_category + pressure_category + is_precipitation +
       weather_season + dep_region

NOTE:
    3-month scope. Seasonal output will only show Winter and Spring.
    All column names lowercase to match Silver layer convention.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from configs.configs import SILVER_BASE, GOLD_BASE

SILVER_FLIGHTS_PATH  = f"{SILVER_BASE}/flights/"
SILVER_WEATHER_PATH  = f"{SILVER_BASE}/weather/"
SILVER_AIRPORTS_PATH = f"{SILVER_BASE}/airports/"
GOLD_PATH            = f"{GOLD_BASE}/weather_delay_impact/"


def build_weather_delay_impact(spark: SparkSession) -> None:

    print("=" * 60)
    print("Gold | weather_delay_impact | Starting")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Read Silver tables
    # ------------------------------------------------------------------
    flights  = spark.read.parquet(SILVER_FLIGHTS_PATH)
    weather  = spark.read.parquet(SILVER_WEATHER_PATH)
    airports = spark.read.parquet(SILVER_AIRPORTS_PATH)
    flights.cache()
    weather.cache()
    airports.cache()
    flights.count()
    weather.count()
    airports.count()
    print(f"[Silver] Flights rows        : {flights.count():,}")
    print(f"[Silver] Weather rows        : {weather.count():,}")
    print(f"[Silver] Airports rows       : {airports.count():,}")

    # ------------------------------------------------------------------
    # 2. Slim weather to only needed columns — avoids name collisions
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
    # 3. LEFT JOIN flights → weather (flights is anchor)
    # ------------------------------------------------------------------
    joined = flights.join(
        weather_slim,
        (flights["dep_airport"] == weather_slim["iata_code"]) &
        (flights["flightdate"]  == weather_slim["weather_date"]),
        how="left"
    ).drop("iata_code", "weather_date")

    # ------------------------------------------------------------------
    # 4. Broadcast join departure airports
    # ------------------------------------------------------------------
    dep_airports = airports.select(
        F.col("iata_code"),
        F.col("us_region").alias("dep_region"),
    )
    joined = joined.join(
        F.broadcast(dep_airports),
        joined["dep_airport"] == dep_airports["iata_code"],
        how="left"
    ).drop("iata_code")

    # ------------------------------------------------------------------
    # 5. Weather join coverage check
    # ------------------------------------------------------------------
    total     = joined.count()
    matched   = joined.filter(F.col("wind_category").isNotNull()).count()
    match_pct = round(matched / total * 100, 1) if total > 0 else 0
    print(f"\n[Join] Total flight rows     : {total:,}")
    print(f"[Join] Matched to weather    : {matched:,}  ({match_pct}%)")
    print(f"[Join] Unmatched             : {total - matched:,}")

    # ------------------------------------------------------------------
    # 6. Aggregate
    # ------------------------------------------------------------------
    delayed_flag = (F.col("dep_delay") > 0).cast("int")
    has_weather  = F.col("wind_category").isNotNull().cast("int")

    gold_df = joined.groupBy(
        "weather_season",
        "wind_category",
        "pressure_category",
        "is_precipitation",
        "dep_region",
    ).agg(
        F.count("*")                                      .alias("total_flights"),
        F.sum(has_weather)                                .alias("flights_with_weather"),
        F.round(F.avg(has_weather) * 100, 1)              .alias("weather_match_pct"),
        F.round(F.avg("dep_delay"), 1)                    .alias("avg_dep_delay_min"),
        F.round(F.avg("arr_delay"), 1)                    .alias("avg_arr_delay_min"),
        F.round(F.avg(delayed_flag) * 100, 1)             .alias("delayed_flight_pct"),
        F.round(F.avg("total_delay_minutes"), 1)          .alias("avg_total_delay_min"),
        F.round(F.avg("temp_avg_c"), 1)                   .alias("avg_temp_c"),
        F.round(F.avg("temp_range_c"), 1)                 .alias("avg_temp_range_c"),
        F.round(F.avg("wind_speed_kmh"), 1)               .alias("avg_wind_speed_kmh"),
        F.round(F.avg("precipitation_mm"), 2)             .alias("avg_precipitation_mm"),
        F.round(F.avg("snowfall_mm"), 2)                  .alias("avg_snowfall_mm"),
    )

    # ------------------------------------------------------------------
    # 7. Sanity checks
    # ------------------------------------------------------------------
    gold_count = gold_df.count()
    print(f"\n[Gold] Output rows           : {gold_count:,}")
    print(f"[Gold] Output columns        : {len(gold_df.columns)}")

    print(f"\n[Gold] Avg dep delay by wind_category:")
    gold_df.groupBy("wind_category") \
           .agg(
               F.round(F.avg("avg_dep_delay_min"), 1).alias("avg_dep_delay"),
               F.round(F.avg("delayed_flight_pct"), 1).alias("delayed_pct"),
               F.sum("total_flights").alias("total_flights"),
           ).orderBy("wind_category").show(truncate=False)

    print(f"[Gold] Avg dep delay by pressure_category:")
    gold_df.groupBy("pressure_category") \
           .agg(
               F.round(F.avg("avg_dep_delay_min"), 1).alias("avg_dep_delay"),
               F.round(F.avg("delayed_flight_pct"), 1).alias("delayed_pct"),
           ).orderBy("pressure_category").show(truncate=False)

    print(f"[Gold] Precipitation vs no precipitation:")
    gold_df.groupBy("is_precipitation") \
           .agg(
               F.round(F.avg("avg_dep_delay_min"), 1).alias("avg_dep_delay"),
               F.round(F.avg("delayed_flight_pct"), 1).alias("delayed_pct"),
               F.sum("total_flights").alias("total_flights"),
           ).orderBy("is_precipitation").show(truncate=False)

    # ------------------------------------------------------------------
    # 8. Write Gold parquet
    # ------------------------------------------------------------------
    gold_df.write \
           .mode("overwrite") \
           .parquet(GOLD_PATH)

    print(f"\n[Gold] Written to            : {GOLD_PATH}")
    flights.unpersist()
    weather.unpersist()
    airports.unpersist()
    print("=" * 60)
    print("Gold | weather_delay_impact | Complete ✓")
    print("=" * 60)


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("Gold_WeatherDelayImpact") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    build_weather_delay_impact(spark)
    spark.stop()