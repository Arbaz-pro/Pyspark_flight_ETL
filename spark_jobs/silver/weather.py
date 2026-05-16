"""
silver/weather.py
------------------
Silver layer transformation for the weather_meteo_by_airport dataset.

Reads from:
    {BRONZE_BASE}/weather/

Writes to:
    {SILVER_BASE}/weather/
    Partitioned by: Weather_Year, Weather_Month

Transformations applied:
    1.  Remove exact duplicates
    2.  Rename columns to readable standard
            time       → Weather_Date
            airport_id → IATA_CODE          (join key alignment with airports/flights)
            tavg       → Temp_Avg_C
            tmin       → Temp_Min_C
            tmax       → Temp_Max_C
            prcp       → Precipitation_mm
            snow       → Snowfall_mm
            wdir       → Wind_Direction_deg
            wspd       → Wind_Speed_kmh
            pres       → Pressure_hPa
    3.  Null handling
            snow null  → 0.0  (null means no snow event, not missing data)
            wdir null when wspd == 0 → already null (calm wind has no direction — correct)
            wdir null when wspd  > 0 → keep null  (genuinely missing reading)
    4.  Extract Weather_Year, Weather_Month, Weather_Day from Weather_Date
    5.  Derive Weather_Season from month
    6.  Derive Temp_Range_C (tmax - tmin — daily temperature swing)
    7.  Derive Is_Precipitation (boolean — prcp > 0)
    8.  Derive Wind_Category from Wind_Speed_kmh buckets
    9.  Derive Pressure_Category from Pressure_hPa
    10. Add Silver_Load_Timestamp

JOIN KEY FOR GOLD:
    IATA_CODE + Weather_Date → matches Dep_Airport + FlightDate in flights Silver
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from configs.configs import BRONZE_BASE, SILVER_BASE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BRONZE_PATH = f"{BRONZE_BASE}/weather/"
SILVER_PATH = f"{SILVER_BASE}/weather/"


# ---------------------------------------------------------------------------
# Helper: Wind_Category from Wind_Speed_kmh
# Beaufort-inspired buckets, practical for flight delay analysis
# ---------------------------------------------------------------------------

def add_wind_category(df):
    """
    Calm     :  wspd == 0
    Light    :  0  < wspd <= 20
    Moderate :  20 < wspd <= 40
    Strong   :  40 < wspd <= 60
    Severe   :  wspd > 60
    """
    return df.withColumn(
        "Wind_Category",
        F.when(F.col("Wind_Speed_kmh").isNull(), None)
         .when(F.col("Wind_Speed_kmh") == 0,    "Calm")
         .when(F.col("Wind_Speed_kmh") <= 20,   "Light")
         .when(F.col("Wind_Speed_kmh") <= 40,   "Moderate")
         .when(F.col("Wind_Speed_kmh") <= 60,   "Strong")
         .otherwise("Severe")
    )


# ---------------------------------------------------------------------------
# Helper: Weather_Season from month number
# ---------------------------------------------------------------------------

def add_weather_season(df):
    """
    Winter : Dec, Jan, Feb  (months 12, 1, 2)
    Spring : Mar, Apr, May
    Summer : Jun, Jul, Aug
    Fall   : Sep, Oct, Nov
    """
    return df.withColumn(
        "Weather_Season",
        F.when(F.col("Weather_Month").isin(12, 1, 2), "Winter")
         .when(F.col("Weather_Month").isin(3, 4, 5),  "Spring")
         .when(F.col("Weather_Month").isin(6, 7, 8),  "Summer")
         .when(F.col("Weather_Month").isin(9, 10, 11),"Fall")
         .otherwise(None)
    )


# ---------------------------------------------------------------------------
# Helper: Pressure_Category from Pressure_hPa
# Useful as a proxy for storm/weather system presence
# ---------------------------------------------------------------------------

def add_pressure_category(df):
    """
    Low      : pres < 1000 hPa  (storm / low-pressure system)
    Normal   : 1000 <= pres <= 1020
    High     : pres > 1020      (high-pressure, generally clear)
    """
    return df.withColumn(
        "Pressure_Category",
        F.when(F.col("Pressure_hPa").isNull(), None)
         .when(F.col("Pressure_hPa") < 1000,  "Low")
         .when(F.col("Pressure_hPa") <= 1020, "Normal")
         .otherwise("High")
    )


# ---------------------------------------------------------------------------
# Main transformation function
# ---------------------------------------------------------------------------

def transform_weather(spark: SparkSession) -> None:

    print("=" * 60)
    print("Silver | weather | Starting transformation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Read Bronze parquet
    # ------------------------------------------------------------------
    df = spark.read.parquet(BRONZE_PATH)
    df.cache()
    bronze_count = df.count()
    print(f"[Bronze] Rows read           : {bronze_count:,}")

    # ------------------------------------------------------------------
    # 2. Remove exact duplicates
    # ------------------------------------------------------------------
    df = df.dropDuplicates()
    dedup_count = df.count()
    print(f"[Dedup] Rows after dedup     : {dedup_count:,}")
    print(f"[Dedup] Duplicates removed   : {bronze_count - dedup_count:,}")

    # ------------------------------------------------------------------
    # 3. Rename columns to readable standard
    # ------------------------------------------------------------------
    df = df.withColumnRenamed("time",       "Weather_Date") \
           .withColumnRenamed("airport_id", "IATA_CODE") \
           .withColumnRenamed("tavg",       "Temp_Avg_C") \
           .withColumnRenamed("tmin",       "Temp_Min_C") \
           .withColumnRenamed("tmax",       "Temp_Max_C") \
           .withColumnRenamed("prcp",       "Precipitation_mm") \
           .withColumnRenamed("snow",       "Snowfall_mm") \
           .withColumnRenamed("wdir",       "Wind_Direction_deg") \
           .withColumnRenamed("wspd",       "Wind_Speed_kmh") \
           .withColumnRenamed("pres",       "Pressure_hPa")

    # ------------------------------------------------------------------
    # 4. Null handling
    #    snow: null means no snowfall event → 0.0
    #    wdir: null when wspd=0 is correct (calm wind has no direction)
    #          null when wspd>0 is a genuine missing reading — keep null
    # ------------------------------------------------------------------
    df = df.withColumn(
        "Snowfall_mm",
        F.coalesce(F.col("Snowfall_mm"), F.lit(0.0))
    )

    # ------------------------------------------------------------------
    # 5. Extract date parts from Weather_Date (already DateType from Bronze)
    # ------------------------------------------------------------------
    df = df.withColumn("Weather_Year",  F.year("Weather_Date").cast(IntegerType()))
    df = df.withColumn("Weather_Month", F.month("Weather_Date").cast(IntegerType()))
    df = df.withColumn("Weather_Day",   F.dayofmonth("Weather_Date").cast(IntegerType()))

    # ------------------------------------------------------------------
    # 6. Derive Weather_Season
    # ------------------------------------------------------------------
    df = add_weather_season(df)

    # ------------------------------------------------------------------
    # 7. Derive Temp_Range_C — daily temperature swing
    #    High range = unstable day, useful ML feature for delay prediction
    # ------------------------------------------------------------------
    df = df.withColumn(
        "Temp_Range_C",
        F.when(
            F.col("Temp_Max_C").isNotNull() & F.col("Temp_Min_C").isNotNull(),
            F.round(F.col("Temp_Max_C") - F.col("Temp_Min_C"), 1)
        ).otherwise(None)
    )

    # ------------------------------------------------------------------
    # 8. Derive Is_Precipitation — boolean flag
    #    Cleaner than checking prcp > 0 in every Gold query
    # ------------------------------------------------------------------
    df = df.withColumn(
        "Is_Precipitation",
        F.when(F.col("Precipitation_mm").isNull(), None)
         .when(F.col("Precipitation_mm") > 0, True)
         .otherwise(False)
    )

    # ------------------------------------------------------------------
    # 9. Derive Wind_Category
    # ------------------------------------------------------------------
    df = add_wind_category(df)

    # ------------------------------------------------------------------
    # 10. Derive Pressure_Category
    # ------------------------------------------------------------------
    df = add_pressure_category(df)

    # ------------------------------------------------------------------
    # 11. Uppercase IATA_CODE for join safety
    # ------------------------------------------------------------------
    df = df.withColumn("IATA_CODE", F.upper(F.col("IATA_CODE")))

    # ------------------------------------------------------------------
    # 12. Silver_Load_Timestamp
    # ------------------------------------------------------------------
    df = df.withColumn("Silver_Load_Timestamp", F.current_timestamp())

    # ------------------------------------------------------------------
    # 13. Final column ordering
    # ------------------------------------------------------------------
    final_cols = [
        # Join keys
        "IATA_CODE", "Weather_Date",
        "Weather_Year", "Weather_Month", "Weather_Day", "Weather_Season",

        # Temperature
        "Temp_Avg_C", "Temp_Min_C", "Temp_Max_C", "Temp_Range_C",

        # Precipitation and snow
        "Precipitation_mm", "Is_Precipitation", "Snowfall_mm",

        # Wind
        "Wind_Speed_kmh", "Wind_Direction_deg", "Wind_Category",

        # Pressure
        "Pressure_hPa", "Pressure_Category",

        # Metadata
        "Silver_Load_Timestamp",
    ]

    df = df.select(*final_cols)

    # ------------------------------------------------------------------
    # 14. Sanity checks before write
    # ------------------------------------------------------------------
    silver_count = df.count()
    print(f"\n[Silver] Rows to write       : {silver_count:,}")
    print(f"[Silver] Output columns      : {len(df.columns)}")

    print(f"\n[Silver] Weather_Season distribution:")
    df.groupBy("Weather_Season").count().orderBy("Weather_Season").show()

    print(f"[Silver] Wind_Category distribution:")
    df.groupBy("Wind_Category").count().orderBy("Wind_Category").show()

    print(f"[Silver] Pressure_Category distribution:")
    df.groupBy("Pressure_Category").count().orderBy("Pressure_Category").show()

    print(f"[Silver] Is_Precipitation distribution:")
    df.groupBy("Is_Precipitation").count().orderBy("Is_Precipitation").show()

    print(f"[Silver] Null counts on key columns:")
    df.select(
        F.count(F.when(F.col("Temp_Avg_C").isNull(),         1)).alias("Temp_Avg_Nulls"),
        F.count(F.when(F.col("Precipitation_mm").isNull(),   1)).alias("Precip_Nulls"),
        F.count(F.when(F.col("Wind_Speed_kmh").isNull(),     1)).alias("Wind_Speed_Nulls"),
        F.count(F.when(F.col("Wind_Direction_deg").isNull(), 1)).alias("Wind_Dir_Nulls"),
        F.count(F.when(F.col("Pressure_hPa").isNull(),       1)).alias("Pressure_Nulls"),
        F.count(F.when(F.col("Snowfall_mm").isNull(),        1)).alias("Snow_Nulls"),
    ).show()

    # ------------------------------------------------------------------
    # 15. Write Silver parquet partitioned by Weather_Year, Weather_Month
    # ------------------------------------------------------------------
    df = df.toDF(*[c.lower() for c in df.columns])
    df.write \
      .mode("overwrite") \
      .partitionBy("weather_year", "weather_month") \
      .parquet(SILVER_PATH)

    print(f"\n[Silver] Written to          : {SILVER_PATH}")
    df.unpersist()
    print(f"[Silver] Partitioned by      : Weather_Year, Weather_Month")
    print("=" * 60)
    print("Silver | weather | Transformation complete ✓")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------
