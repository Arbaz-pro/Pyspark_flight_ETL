"""
gold/flight_performance.py
---------------------------
Gold layer — Flight Performance Summary

Sources (Silver):
    - flights   (3 months — Jan/Feb/Mar 2023)
    - airports  (static, 364 rows — broadcast joined)

Output:
    /home/asus/data_lake/gold/flight_performance/
    Partitioned by: flight_year, flight_month

Grain: One row per airline + dep_airport + arr_airport + flight_month + distance_type

All column names lowercase to match Silver layer convention.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from configs.configs import SILVER_BASE, GOLD_BASE


SILVER_FLIGHTS_PATH  = f"{SILVER_BASE}/flights/"
SILVER_AIRPORTS_PATH = f"{SILVER_BASE}/airports/"
GOLD_PATH            = f"{GOLD_BASE}/flight_performance/"


def build_flight_performance(spark: SparkSession) -> None:

    print("=" * 60)
    print("Gold | flight_performance | Starting")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Read Silver tables
    # ------------------------------------------------------------------
    flights  = spark.read.parquet(SILVER_FLIGHTS_PATH)
    airports = spark.read.parquet(SILVER_AIRPORTS_PATH)
    flights.cache()
    airports.cache()
    flights.count()
    airports.count()
    print(f"[Silver] Flights rows        : {flights.count():,}")
    print(f"[Silver] Airports rows       : {airports.count():,}")

    # ------------------------------------------------------------------
    # 2. Broadcast join airports onto departure side
    #    Airports is 364 rows — always broadcast, never shuffle
    # ------------------------------------------------------------------
    dep_airports = airports.select(
        F.col("iata_code"),
        F.col("city").alias("dep_city"),
        F.col("state").alias("dep_state"),
        F.col("us_region").alias("dep_region"),
    )

    flights = flights.join(
        F.broadcast(dep_airports),
        flights["dep_airport"] == dep_airports["iata_code"],
        how="left"
    ).drop("iata_code")

    # ------------------------------------------------------------------
    # 3. Aggregate — grain: airline + route + month + distance_type
    # ------------------------------------------------------------------
    delayed_flag  = (F.col("dep_delay") > 0).cast("int")
    on_time_flag  = (F.col("dep_delay") <= 0).cast("int")
    carrier_flag  = (F.col("primary_delay_cause") == "Carrier").cast("int")
    weather_flag  = (F.col("primary_delay_cause") == "Weather").cast("int")
    nas_flag      = (F.col("primary_delay_cause") == "NAS").cast("int")
    aircraft_flag = (F.col("primary_delay_cause") == "LastAircraft").cast("int")

    gold_df = flights.groupBy(
        "flight_year", "flight_month",
        "airline",
        "dep_airport", "dep_city", "dep_state", "dep_region",
        "arr_airport", "arr_cityname",
        "distance_type",
    ).agg(
        F.count("*")                                     .alias("total_flights"),
        F.round(F.avg("dep_delay"), 1)                   .alias("avg_dep_delay_min"),
        F.round(F.avg("arr_delay"), 1)                   .alias("avg_arr_delay_min"),
        F.max("dep_delay")                               .alias("max_dep_delay_min"),
        F.round(F.avg("flight_duration"), 1)             .alias("avg_flight_duration_min"),
        F.round(F.avg("total_delay_minutes"), 1)         .alias("avg_total_delay_min"),
        F.round(F.avg(on_time_flag) * 100, 1)            .alias("on_time_pct"),
        F.round(F.avg(delayed_flag) * 100, 1)            .alias("delayed_pct"),
        F.round(F.avg(carrier_flag)  * 100, 1)           .alias("carrier_delay_pct"),
        F.round(F.avg(weather_flag)  * 100, 1)           .alias("weather_delay_pct"),
        F.round(F.avg(nas_flag)      * 100, 1)           .alias("nas_delay_pct"),
        F.round(F.avg(aircraft_flag) * 100, 1)           .alias("aircraft_delay_pct"),
    )

    # ------------------------------------------------------------------
    # 4. Sanity checks
    # ------------------------------------------------------------------
    gold_count = gold_df.count()
    print(f"\n[Gold] Output rows           : {gold_count:,}")
    print(f"[Gold] Output columns        : {len(gold_df.columns)}")

    print(f"\n[Gold] Top 10 routes by avg dep delay:")
    gold_df.orderBy(F.col("avg_dep_delay_min").desc()).show(10, truncate=False)

    print(f"[Gold] On-time % by airline:")
    gold_df.groupBy("airline") \
           .agg(F.round(F.avg("on_time_pct"), 1).alias("avg_on_time_pct")) \
           .orderBy(F.col("avg_on_time_pct").desc()).show(15, truncate=False)

    print(f"[Gold] Performance by region:")
    gold_df.groupBy("dep_region") \
           .agg(
               F.round(F.avg("avg_dep_delay_min"), 1).alias("avg_dep_delay"),
               F.round(F.avg("on_time_pct"), 1).alias("on_time_pct"),
               F.sum("total_flights").alias("total_flights"),
           ).orderBy("dep_region").show(truncate=False)

    # ------------------------------------------------------------------
    # 5. Write Gold parquet
    # ------------------------------------------------------------------
    gold_df.write \
           .mode("overwrite") \
           .partitionBy("flight_year", "flight_month") \
           .parquet(GOLD_PATH)

    print(f"\n[Gold] Written to            : {GOLD_PATH}")
    airports.unpersist()
    flights.unpersist()
    print("=" * 60)
    print("Gold | flight_performance | Complete ✓")
    print("=" * 60)


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("Gold_FlightPerformance") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    build_flight_performance(spark)
    spark.stop()