"""
silver/flights.py
------------------
Silver layer transformation for the US_flights_2023 dataset.

Reads from:
    /home/asus/data_lake/bronze/flights/

Writes to:
    /home/asus/data_lake/silver/flights/
    Partitioned by: Flight_Year, Flight_Month

Transformations applied:
    1.  Remove exact duplicates
    2.  Rename typo column: Aicraft_age → Aircraft_age
    3.  Trim all string columns
    4.  Uppercase airport / manufacturer columns
    5.  Fix "Hight" typo → "High" in delay type columns
    6.  Standardize Distance_type labels (exact match, no .contains())
    7.  Derive Total_Delay_Minutes (null-safe; null if all causes are null)
    8.  Derive Primary_Delay_Cause
    9.  Derive Departure_Status and Arrival_Status (null-safe)
    10. Parse FlightDate string (dd-MM-yyyy) → DateType
    11. Extract Flight_Year, Flight_Month, Flight_Day, Weekday_Name
    12. Final column ordering via select()
    13. Add Silver_Load_Timestamp
    14. Write partitioned parquet

DISTANCE_TYPE STANDARD (agreed across all Silver jobs):
    "Short Haul"    (was "Short Haul >1500Mi" in Bronze)
    "Medium Haul"   (was "Medium Haul <3000Mi" in Bronze)
    "Long Haul"     (was "Long Haul <6000Mi"   in Bronze)
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from configs.configs import BRONZE_BASE, SILVER_BASE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BRONZE_PATH = f"{BRONZE_BASE}/flights/"
SILVER_PATH = f"{SILVER_BASE}/flights/"

DELAY_CAUSE_COLS = [
    "Delay_Carrier",
    "Delay_Weather",
    "Delay_NAS",
    "Delay_Security",
    "Delay_LastAircraft",
]


# ---------------------------------------------------------------------------
# Helper: Total_Delay_Minutes — null-safe
# ---------------------------------------------------------------------------

def add_total_delay(df):
    """
    Sums the 5 delay cause columns.

    If ALL five columns are null (data gap, not a zero-delay flight),
    returns null rather than a misleading 0.
    If at least one is non-null, treats remaining nulls as 0.
    """
    all_null_condition = (
        F.col("Delay_Carrier").isNull() &
        F.col("Delay_Weather").isNull() &
        F.col("Delay_NAS").isNull() &
        F.col("Delay_Security").isNull() &
        F.col("Delay_LastAircraft").isNull()
    )
    total_expr = sum(
        F.coalesce(F.col(c), F.lit(0)) for c in DELAY_CAUSE_COLS
    )
    return df.withColumn(
        "Total_Delay_Minutes",
        F.when(all_null_condition, None)
         .otherwise(total_expr.cast(IntegerType()))
    )


# ---------------------------------------------------------------------------
# Helper: Primary_Delay_Cause
# ---------------------------------------------------------------------------

def add_primary_delay_cause(df):
    """
    Returns the dominant delay cause by highest minute contribution.
    - Total_Delay_Minutes == 0 → "None"
    - Total_Delay_Minutes is null → null
    Ties broken by column order (Carrier > Weather > NAS > Security > LastAircraft).
    """
    cause_expr = (
        F.when(F.col("Total_Delay_Minutes").isNull(), None)
         .when(F.col("Total_Delay_Minutes") == 0, "None")
         .when(
             (F.col("Delay_Carrier") >= F.col("Delay_Weather")) &
             (F.col("Delay_Carrier") >= F.col("Delay_NAS")) &
             (F.col("Delay_Carrier") >= F.col("Delay_Security")) &
             (F.col("Delay_Carrier") >= F.col("Delay_LastAircraft")),
             "Carrier"
         )
         .when(
             (F.col("Delay_Weather") >= F.col("Delay_NAS")) &
             (F.col("Delay_Weather") >= F.col("Delay_Security")) &
             (F.col("Delay_Weather") >= F.col("Delay_LastAircraft")),
             "Weather"
         )
         .when(
             (F.col("Delay_NAS") >= F.col("Delay_Security")) &
             (F.col("Delay_NAS") >= F.col("Delay_LastAircraft")),
             "NAS"
         )
         .when(
             F.col("Delay_Security") >= F.col("Delay_LastAircraft"),
             "Security"
         )
         .otherwise("LastAircraft")
    )
    return df.withColumn("Primary_Delay_Cause", cause_expr)


# ---------------------------------------------------------------------------
# Helper: Departure and Arrival Status — null-safe
# ---------------------------------------------------------------------------

def add_flight_statuses(df):
    """
    Classifies departure and arrival timing.
    Null delay values return null status rather than a misleading "Delayed".
    """
    df = df.withColumn(
        "Departure_Status",
        F.when(F.col("Dep_Delay").isNull(), None)
         .when(F.col("Dep_Delay") < 0,  "Early")
         .when(F.col("Dep_Delay") == 0, "On Time")
         .otherwise("Delayed")
    )
    df = df.withColumn(
        "Arrival_Status",
        F.when(F.col("Arr_Delay").isNull(), None)
         .when(F.col("Arr_Delay") < 0,  "Early")
         .when(F.col("Arr_Delay") == 0, "On Time")
         .otherwise("Delayed")
    )
    return df


# ---------------------------------------------------------------------------
# Main transformation function
# ---------------------------------------------------------------------------

def transform_flights_silver(spark: SparkSession) -> None:

    print("=" * 60)
    print("Silver | flights | Starting transformation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Read Bronze parquet
    # ------------------------------------------------------------------
    BRONZE_PATH = "/home/asus/data_lake/bronze/flights/"
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
    # 3. Rename typo column
    # ------------------------------------------------------------------
    df = df.withColumnRenamed("Aicraft_age", "Aircraft_age")

    # ------------------------------------------------------------------
    # 4. Trim all string columns
    # ------------------------------------------------------------------
    str_cols = [
        field.name for field in df.schema.fields
        if str(field.dataType) == "StringType()"
    ]
    for col_name in str_cols:
        df = df.withColumn(col_name, F.trim(F.col(col_name)))

    # ------------------------------------------------------------------
    # 5. Uppercase airport and manufacturer columns
    # ------------------------------------------------------------------
    for col_name in ("Dep_Airport", "Arr_Airport", "Manufacturer"):
        df = df.withColumn(col_name, F.upper(F.col(col_name)))

    # ------------------------------------------------------------------
    # 6. Fix "Hight" typo in delay type columns
    #    "Hight >60min" → "High >60min"
    # ------------------------------------------------------------------
    for col_name in ("Dep_Delay_Type", "Arr_Delay_Type"):
        df = df.withColumn(
            col_name,
            F.regexp_replace(F.col(col_name), r"(?i)Hight", "High")
        )

    # ------------------------------------------------------------------
    # 7. Standardize Distance_type labels
    #    Exact match — NOT .contains() which is fragile against label changes.
    #    Standard agreed across all Silver jobs:
    #      "Short Haul" / "Medium Haul" / "Long Haul"
    # ------------------------------------------------------------------
    df = df.withColumn(
        "Distance_type",
        F.when(F.col("Distance_type") == "Short Haul >1500Mi",  "Short Haul")
         .when(F.col("Distance_type") == "Medium Haul <3000Mi", "Medium Haul")
         .when(F.col("Distance_type") == "Long Haul <6000Mi",   "Long Haul")
         .otherwise(F.col("Distance_type"))   # pass-through unknown labels
    )

    # ------------------------------------------------------------------
    # 8. Total_Delay_Minutes — null-safe sum of 5 cause columns
    # ------------------------------------------------------------------
    df = add_total_delay(df)

    # ------------------------------------------------------------------
    # 9. Primary_Delay_Cause
    # ------------------------------------------------------------------
    df = add_primary_delay_cause(df)

    # ------------------------------------------------------------------
    # 10. Departure_Status and Arrival_Status — null-safe
    # ------------------------------------------------------------------
    df = add_flight_statuses(df)

    # ------------------------------------------------------------------
    # 11. Parse FlightDate string → DateType, then extract date parts
    #     Original string format: dd-MM-yyyy
    #     FlightDate is overwritten with the parsed DateType (cleaner for
    #     downstream joins against weather/cancelled which use DateType)
    # ------------------------------------------------------------------
    df = df.withColumn(
        "FlightDate",
        F.to_date(F.col("FlightDate"), "dd-MM-yyyy")
    )
    df = df.withColumn("Flight_Year",  F.year("FlightDate").cast(IntegerType()))
    df = df.withColumn("Flight_Month", F.month("FlightDate").cast(IntegerType()))
    df = df.withColumn("Flight_Day",   F.dayofmonth("FlightDate").cast(IntegerType()))
    df = df.withColumn("Weekday_Name", F.date_format("FlightDate", "EEEE"))

    # ------------------------------------------------------------------
    # 12. Silver_Load_Timestamp
    # ------------------------------------------------------------------
    df = df.withColumn("Silver_Load_Timestamp", F.current_timestamp())

    # ------------------------------------------------------------------
    # 13. Final column ordering via explicit select()
    #     Consistent grouping makes downstream Gold joins readable
    # ------------------------------------------------------------------
    final_cols = [
        # Identifiers
        "FlightDate", "Flight_Year", "Flight_Month", "Flight_Day",
        "Day_Of_Week", "Weekday_Name",
        "Airline", "Tail_Number",

        # Aircraft
        "Manufacturer", "Model", "Aircraft_age",

        # Route
        "Dep_Airport", "Dep_CityName",
        "Arr_Airport", "Arr_CityName",

        # Timing and distance
        "DepTime_label", "Flight_Duration", "Distance_type",

        # Departure delay
        "Dep_Delay", "Dep_Delay_Tag", "Dep_Delay_Type", "Departure_Status",

        # Arrival delay
        "Arr_Delay", "Arr_Delay_Type", "Arrival_Status",

        # Delay breakdown
        "Delay_Carrier", "Delay_Weather", "Delay_NAS",
        "Delay_Security", "Delay_LastAircraft",

        # Derived delay summary
        "Total_Delay_Minutes", "Primary_Delay_Cause",

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

    print(f"\n[Silver] Departure_Status distribution:")
    df.groupBy("Departure_Status").count().orderBy("Departure_Status").show()

    print(f"[Silver] Arrival_Status distribution:")
    df.groupBy("Arrival_Status").count().orderBy("Arrival_Status").show()

    print(f"[Silver] Primary_Delay_Cause distribution:")
    df.groupBy("Primary_Delay_Cause").count() \
      .orderBy(F.col("count").desc()).show()

    print(f"[Silver] Distance_type distribution (verify standardization):")
    df.groupBy("Distance_type").count().orderBy("Distance_type").show()

    # ------------------------------------------------------------------
    # 15. Write Silver parquet partitioned by Flight_Year, Flight_Month
    # ------------------------------------------------------------------
    df = df.toDF(*[c.lower() for c in df.columns])
    df.write \
        .mode("overwrite") \
        .partitionBy("flight_year", "flight_month") \
        .parquet(SILVER_PATH)

    print(f"\n[Silver] Written to          : {SILVER_PATH}")
    df.unpersist()
    print(f"[Silver] Partitioned by      : Flight_Year, Flight_Month")
    print("=" * 60)
    print("Silver | flights | Transformation complete ✓")
    print("=" * 60)

