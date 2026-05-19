from pyspark.sql import functions as F
from configs.configs import BRONZE_BASE, RAW_BASE
from spark_jobs.checks import profile_dataframe

def ingest_flights(spark):

    df = spark.read.csv(
        f"{RAW_BASE}/US_flights_2023.csv",
        header=True,
        inferSchema=True
    )
    # profile_dataframe(df)

    df.write.mode("overwrite").parquet(
        f"{BRONZE_BASE}/flights/"
    )

    print("\nFlights ingestion completed")