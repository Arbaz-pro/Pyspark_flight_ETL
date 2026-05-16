from spark_jobs.checks import profile_dataframe
from configs.configs import BRONZE_BASE, RAW_BASE

def ingest_weather(spark):
    
    df=spark.read.csv(
        f"{RAW_BASE}/weather_meteo_by_airport.csv",
        header=True,
        inferSchema=True
    )

    # profile_dataframe(df)

    df.write.mode("overwrite").parquet(
        f"{BRONZE_BASE}/weather/"
        )
    print("Weather ingestion completed")  