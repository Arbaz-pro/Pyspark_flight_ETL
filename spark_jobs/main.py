from pyspark.sql import SparkSession

from spark_jobs.bronze.airports_ingestion  import ingest_airports
from spark_jobs.bronze.cancelled_ingestion import ingest_cancelled
from spark_jobs.bronze.flights_ingestion   import ingest_flights
from spark_jobs.bronze.weather_ingestion   import ingest_weather

from spark_jobs.silver.flights    import transform_flights_silver
from spark_jobs.silver.cancelled  import transform_cancelled
from spark_jobs.silver.weather    import transform_weather
from spark_jobs.silver.airport    import transform_airport

from spark_jobs.gold.flight_performance    import build_flight_performance
from spark_jobs.gold.weather_delay_impact  import build_weather_delay_impact
from spark_jobs.gold.cancellation_analysis import build_cancellation_analysis
from spark_jobs.gold.weather_cancellation  import build_weather_cancellation


if __name__ == "__main__":

    spark = SparkSession.builder \
    .appName("Flight ETL Pipeline") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config(
        "spark.jars.packages",
        "org.apache.hadoop:hadoop-aws:3.3.4"
    ) \
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.DefaultAWSCredentialsProviderChain"
    ) \
    .config(
        "spark.hadoop.fs.s3a.impl",
        "org.apache.hadoop.fs.s3a.S3AFileSystem"
    ) \
    .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # ------------------------------------------------------------------
    # BRONZE
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BRONZE LAYER")
    print("=" * 60)
    ingest_flights(spark)
    ingest_weather(spark)
    ingest_airports(spark)
    ingest_cancelled(spark)
    print("All datasets ingested successfully")

# #     # ------------------------------------------------------------------
# #     # SILVER
# #     # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SILVER LAYER")
    print("=" * 60)
    transform_flights_silver(spark)
    transform_cancelled(spark)
    transform_weather(spark)
    transform_airport(spark)
    print("All Silver transformations completed successfully")

#     # ------------------------------------------------------------------
#     # GOLD
#     # Track A — Flights-anchored (3 months Jan/Feb/Mar 2023)
#     # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("GOLD LAYER — Track A (Flights, 3 months)")
    print("=" * 60)
    build_flight_performance(spark)      # flights + airports
    build_weather_delay_impact(spark)    # flights + weather + airports

    # ------------------------------------------------------------------
    # GOLD
    # Track B — Cancellation-anchored (12 months full 2023)
    # Richer — do not mix raw counts with Track A outputs
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("GOLD LAYER — Track B (Cancellations, 12 months)")
    print("=" * 60)
    build_cancellation_analysis(spark)   # cancelled + airports
    build_weather_cancellation(spark)    # cancelled + weather + airports
    from spark_jobs.gold.ops_summary_3month import build_ops_summary

# In Gold Track A section:
    build_ops_summary(spark)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE ✓")
    print("=" * 60)

    spark.stop()