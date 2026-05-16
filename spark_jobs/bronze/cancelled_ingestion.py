from spark_jobs.checks import profile_dataframe
from configs.configs import BRONZE_BASE, RAW_BASE

def ingest_cancelled(spark):

    df=spark.read.csv(
        f"{RAW_BASE}/Cancelled_Diverted_2023.csv",
        header=True,
        inferSchema=True
    )

    # profile_dataframe(df)

    df.write.mode("overwrite").parquet(
        f"{BRONZE_BASE}/cancelled_flights/"
    )
    print("Cancelled flights ingestion completed")