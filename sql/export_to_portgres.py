from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("ExportToPostgres") \
    .config(
        "spark.jars",
        "/mnt/d/Pyspark_ETL/jars/postgresql-42.7.3.jar"
    ) \
    .getOrCreate()

url = "jdbc:postgresql://localhost:5432/data_platform"

properties = {
    "user": "spark_user",
    "password": "123",
    "driver": "org.postgresql.Driver"
}

# Flights
df = spark.read.parquet(
    "/home/asus/data_lake/silver/flights/"
)

df.write \
    .mode("overwrite") \
    .jdbc(
        url=url,
        table="silver_flights",
        properties=properties
    )

print("Flights exported")

spark.stop()
