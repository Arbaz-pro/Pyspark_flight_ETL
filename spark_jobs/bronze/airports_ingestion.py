from spark_jobs.checks import profile_dataframe
from configs.configs import BRONZE_BASE, RAW_BASE

def ingest_airports(spark):

    df=spark.read.csv(
        f"{RAW_BASE}/airports_geolocation.csv",
        header=True,
        inferSchema=True)
    
    # profile_dataframe(df)

    df.write.mode("overwrite").parquet( 
        f"{BRONZE_BASE}/airports/")
    
    print("Airports ingestion completed")