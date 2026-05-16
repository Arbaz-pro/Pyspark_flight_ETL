"""
silver/cancelled.py
--------------------
Silver layer transformation for the Cancelled_Diverted_2023 dataset.

Reads from:
    /home/asus/data_lake/bronze/cancelled_flights/

Writes to:
    /home/asus/data_lake/silver/cancelled_flights/
    Partitioned by: Flight_Year, Flight_Month

Transformations applied:
    1.  Remove exact duplicates
    2.  Trim all string columns
    3.  Uppercase airport IATA codes (Dep_Airport, Arr_Airport)
    4.  Fix "Hight" typo → "High" in Dep_Delay_Type and Arr_Delay_Type
    5.  Standardize Distance_type labels to match flights dataset format
    6.  Derive Flight_Status: Cancelled / Diverted
    7.  Nullify delay/duration columns for Cancelled=1 rows (flight never flew)
    8.  Parse FlightDate (date type from Bronze) → Flight_Year, Flight_Month, Flight_Day
    9.  Derive Weekday_Name from Day_Of_Week integer
    10. Derive Total_Delay_Minutes for Diverted rows only (Cancelled rows = null)
    11. Derive Primary_Delay_Cause for Diverted rows only
    12. Add Silver_Load_Timestamp

NULLIFICATION LOGIC:
    Cancelled=1  → Dep_Delay, Arr_Delay, all delay causes,
                   Flight_Duration set to null (inapplicable — flight never flew)
    Diverted=1   → all delay values preserved (flight departed, data is real)

DISTANCE_TYPE STANDARDIZATION:
    Source (Cancelled dataset)   →  Target (aligned to Flights dataset)
    "Short Haul"                 →  "Short Haul >1500Mi"
    "Medium Haul"                →  "Medium Haul <3000Mi"
    "Long Haul"                  →  "Long Haul <6000Mi"
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from configs.configs import BRONZE_BASE, SILVER_BASE


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BRONZE_PATH = f"{BRONZE_BASE}/cancelled_flights/"
SILVER_PATH = f"{SILVER_BASE}/cancelled_flights/"

# Delay cause columns — nullified for Cancelled rows, summed for Diverted rows
DELAY_CAUSE_COLS = [
    "Delay_Carrier",
    "Delay_Weather",
    "Delay_NAS",
    "Delay_Security",
    "Delay_LastAircraft",
]

# All columns that are meaningless (structurally zero) for a cancelled flight
# These will be set to null when Cancelled == 1
NULL_FOR_CANCELLED = [
    "Dep_Delay",
    "Dep_Delay_Tag",
    "Dep_Delay_Type",
    "Arr_Delay",
    "Arr_Delay_Type",
    "Delay_Carrier",
    "Delay_Weather",
    "Delay_NAS",
    "Delay_Security",
    "Delay_LastAircraft",
    "Flight_Duration",
]


# ---------------------------------------------------------------------------
# Helper: Weekday_Name from Day_Of_Week integer (1=Monday)
# ---------------------------------------------------------------------------

def add_weekday_name(df):
    expr = (
        F.when(F.col("Day_Of_Week") == 1, "Monday")
         .when(F.col("Day_Of_Week") == 2, "Tuesday")
         .when(F.col("Day_Of_Week") == 3, "Wednesday")
         .when(F.col("Day_Of_Week") == 4, "Thursday")
         .when(F.col("Day_Of_Week") == 5, "Friday")
         .when(F.col("Day_Of_Week") == 6, "Saturday")
         .when(F.col("Day_Of_Week") == 7, "Sunday")
         .otherwise(None)
    )
    return df.withColumn("Weekday_Name", expr)


# ---------------------------------------------------------------------------
# Helper: Primary_Delay_Cause — only meaningful for Diverted rows
# ---------------------------------------------------------------------------

def add_primary_delay_cause(df):
    """
    Returns the name of the dominant delay cause column.
    - Cancelled rows  → null  (delay cols already nullified above)
    - Diverted rows   → cascade comparison across all 5 cause columns
    - Zero-delay rows → "None"
    Ties broken by column order (Carrier > Weather > NAS > Security > LastAircraft).
    """
    cause_expr = (
        F.when(F.col("Cancelled") == 1, None)
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
# Main transformation function
# ---------------------------------------------------------------------------

def transform_cancelled(spark: SparkSession) -> None:

    print("=" * 60)
    print("Silver | cancelled_flights | Starting transformation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Read Bronze parquet
    #    NOTE: FlightDate ingested as date type in Bronze (unlike flights
    #    dataset which is string). No string-to-date parsing needed here.
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
    # 3. Trim all string columns
    # ------------------------------------------------------------------
    string_cols = [
        field.name for field in df.schema.fields
        if str(field.dataType) == "StringType()"
    ]
    for col_name in string_cols:
        df = df.withColumn(col_name, F.trim(F.col(col_name)))

    # ------------------------------------------------------------------
    # 4. Uppercase airport IATA codes
    # ------------------------------------------------------------------
    df = df.withColumn("Dep_Airport", F.upper(F.col("Dep_Airport")))
    df = df.withColumn("Arr_Airport", F.upper(F.col("Arr_Airport")))

    # ------------------------------------------------------------------
    # 5. Fix "Hight" typo in delay type columns
    #    Cancelled dataset format: "Hight Departure Delay"
    #    After fix:                "High Departure Delay"
    #    (Different format from flights dataset "Hight >60min" but same
    #    root typo — regex handles both)
    # ------------------------------------------------------------------
    for col_name in ("Dep_Delay_Type", "Arr_Delay_Type"):
        df = df.withColumn(
            col_name,
            F.regexp_replace(F.col(col_name), r"(?i)Hight", "High")
        )

    # ------------------------------------------------------------------
    # 6. Standardize Distance_type to match flights dataset labels
    #    Cancelled source  →  Flights dataset standard
    #    "Short Haul"      →  "Short Haul >1500Mi"
    #    "Medium Haul"     →  "Medium Haul <3000Mi"
    #    "Long Haul"       →  "Long Haul <6000Mi"
    #    Critical for Gold GROUP BY across both datasets
    # ------------------------------------------------------------------
    df = df.withColumn(
        "Distance_type",
        F.when(F.col("Distance_type") == "Short Haul",  "Short Haul >1500Mi")
         .when(F.col("Distance_type") == "Medium Haul", "Medium Haul <3000Mi")
         .when(F.col("Distance_type") == "Long Haul",   "Long Haul <6000Mi")
         .otherwise(F.col("Distance_type"))
    )

    # ------------------------------------------------------------------
    # 7. Derive Flight_Status from flags
    #    Cancelled=1 takes precedence over Diverted=1
    # ------------------------------------------------------------------
    df = df.withColumn(
        "Flight_Status",
        F.when(F.col("Cancelled") == 1, "Cancelled")
         .when(F.col("Diverted") == 1, "Diverted")
         .otherwise("Completed")
    )

    # ------------------------------------------------------------------
    # 8. Nullify delay/duration columns for Cancelled=1 rows
    #    Profiling confirmed: Arr_Delay, Flight_Duration, all delay cause
    #    columns have min=max=avg=0.0 — structurally zero, not observed.
    #    Setting to null prevents zeros from polluting Gold averages.
    #    Diverted rows retain real values (Dep_Delay ranges -31 to 2414).
    # ------------------------------------------------------------------
    for col_name in NULL_FOR_CANCELLED:
        df = df.withColumn(
            col_name,
            F.when(F.col("Cancelled") == 1, None)
             .otherwise(F.col(col_name))
        )

    # ------------------------------------------------------------------
    # 9. Extract date parts from FlightDate
    #    Bronze already parsed this as DateType — no string conversion needed
    # ------------------------------------------------------------------
    df = df.withColumn("Flight_Year",  F.year(F.col("FlightDate")).cast(IntegerType()))
    df = df.withColumn("Flight_Month", F.month(F.col("FlightDate")).cast(IntegerType()))
    df = df.withColumn("Flight_Day",   F.dayofmonth(F.col("FlightDate")).cast(IntegerType()))

    # ------------------------------------------------------------------
    # 10. Derive Weekday_Name from Day_Of_Week integer
    # ------------------------------------------------------------------
    df = add_weekday_name(df)

    # ------------------------------------------------------------------
    # 11. Derive Total_Delay_Minutes
    #     Cancelled rows → null (delay cols already null from step 8)
    #     Diverted rows  → sum of 5 cause columns, null-safe coalesce
    # ------------------------------------------------------------------
    total_delay_expr = sum(
        F.coalesce(F.col(c), F.lit(0)) for c in DELAY_CAUSE_COLS
    )
    df = df.withColumn(
        "Total_Delay_Minutes",
        F.when(F.col("Cancelled") == 1, None)
         .otherwise(total_delay_expr.cast(IntegerType()))
    )

    # ------------------------------------------------------------------
    # 12. Derive Primary_Delay_Cause
    # ------------------------------------------------------------------
    df = add_primary_delay_cause(df)

    # ------------------------------------------------------------------
    # 13. Add Silver_Load_Timestamp
    # ------------------------------------------------------------------
    df = df.withColumn("Silver_Load_Timestamp", F.current_timestamp())

    # ------------------------------------------------------------------
    # 14. Final column selection and ordering
    #     Group: identifiers → status → route → timing →
    #            delay info → delay breakdown → derived → metadata
    # ------------------------------------------------------------------
    final_cols = [
        # Identifiers
        "FlightDate", "Flight_Year", "Flight_Month", "Flight_Day",
        "Day_Of_Week", "Weekday_Name",
        "Airline", "Tail_Number",

        # Status flags
        "Cancelled", "Diverted", "Flight_Status",

        # Route
        "Dep_Airport", "Dep_CityName",
        "Arr_Airport", "Arr_CityName",

        # Timing and distance
        "DepTime_label", "Flight_Duration", "Distance_type",

        # Departure delay (null for Cancelled, real for Diverted)
        "Dep_Delay", "Dep_Delay_Tag", "Dep_Delay_Type",

        # Arrival delay (null for Cancelled, real for Diverted)
        "Arr_Delay", "Arr_Delay_Type",

        # Delay breakdown (null for Cancelled, real for Diverted)
        "Delay_Carrier", "Delay_Weather", "Delay_NAS",
        "Delay_Security", "Delay_LastAircraft",

        # Derived delay summary
        "Total_Delay_Minutes", "Primary_Delay_Cause",

        # Metadata
        "Silver_Load_Timestamp",
    ]

    df = df.select(*final_cols)

    # ------------------------------------------------------------------
    # 15. Sanity checks before write
    # ------------------------------------------------------------------
    silver_count = df.count()
    print(f"\n[Silver] Rows to write       : {silver_count:,}")
    print(f"[Silver] Output columns      : {len(df.columns)}")

    print(f"\n[Silver] Flight_Status breakdown:")
    df.groupBy("Flight_Status").count().orderBy("Flight_Status").show()

    print(f"[Silver] Distance_type distribution (verify standardization):")
    df.groupBy("Distance_type").count().orderBy("Distance_type").show()

    print(f"[Silver] Primary_Delay_Cause — Diverted rows only:")
    df.filter(F.col("Diverted") == 1) \
      .groupBy("Primary_Delay_Cause").count() \
      .orderBy(F.col("count").desc()).show()

    print(f"[Silver] Null check — Dep_Delay by Flight_Status:")
    df.groupBy("Flight_Status") \
      .agg(
          F.count(F.when(F.col("Dep_Delay").isNull(), 1)).alias("Dep_Delay_Nulls"),
          F.count(F.when(F.col("Dep_Delay").isNotNull(), 1)).alias("Dep_Delay_Values")
      ).orderBy("Flight_Status").show()
    df = df.toDF(*[c.lower() for c in df.columns])
    # ------------------------------------------------------------------
    # 16. Write Silver parquet partitioned by Flight_Year, Flight_Month
    # ------------------------------------------------------------------
    df.write \
      .mode("overwrite") \
      .partitionBy("flight_year", "flight_month") \
      .parquet(SILVER_PATH)

    print(f"\n[Silver] Written to          : {SILVER_PATH}")
    df.unpersist()
    print(f"[Silver] Partitioned by      : Flight_Year, Flight_Month")
    print("=" * 60)
    print("Silver | cancelled_flights | Transformation complete ✓")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------