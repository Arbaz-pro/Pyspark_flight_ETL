"""
silver/airport.py
------------------
Silver layer transformation for the airports_geolocation dataset.

Reads from:
    /home/asus/data_lake/bronze/airports/

Writes to:
    /home/asus/data_lake/silver/airports/
    No partitioning — 364 rows, static reference table

Transformations applied:
    1.  Remove exact duplicates
    2.  Rename columns to readable snake_case standard
            AIRPORT   → Airport_Name
            CITY      → City
            STATE     → State
            COUNTRY   → Country
            LATITUDE  → Latitude
            LONGITUDE → Longitude
            IATA_CODE → keep as-is (primary join key)
    3.  Trim and uppercase IATA_CODE for join safety
    4.  Trim all other string columns
    5.  Derive US_Region from State
    6.  Derive Is_Alaska_Hawaii flag (weather outliers — very different patterns)
    7.  Derive Hemisphere (N/S — all USA airports are North, but useful for
        any non-US airports that may exist in edge cases)
    8.  Round Latitude and Longitude to 5 decimal places (float precision cleanup)
    9.  Add Silver_Load_Timestamp

JOIN KEY:
    IATA_CODE → Dep_Airport / Arr_Airport in flights Silver
    IATA_CODE → IATA_CODE in weather Silver

NOTE:
    Profiling showed COUNTRY has only 1 unique value: "USA"
    Column retained for schema consistency but noted here.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from configs.configs import BRONZE_BASE, SILVER_BASE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BRONZE_PATH = f"{BRONZE_BASE}/airports/"
SILVER_PATH = f"{SILVER_BASE}/airports/"

# US Region state groupings
# Used for Gold-layer regional aggregations
NE_STATES = ["CT", "ME", "MA", "NH", "RI", "VT", "NY", "NJ", "PA"]
SE_STATES  = ["DE", "MD", "VA", "WV", "NC", "SC", "GA", "FL",
              "AL", "MS", "TN", "KY", "AR", "LA"]
MW_STATES  = ["OH", "MI", "IN", "WI", "IL", "MN", "IA",
              "MO", "ND", "SD", "NE", "KS"]
SW_STATES  = ["TX", "OK", "NM", "AZ"]
W_STATES   = ["CO", "WY", "MT", "ID", "WA", "OR", "CA", "NV", "UT"]
# AK and HI handled separately via Is_Alaska_Hawaii flag


# ---------------------------------------------------------------------------
# Helper: US_Region from State abbreviation
# ---------------------------------------------------------------------------

def add_us_region(df):
    """
    Maps State abbreviation to US geographic region.
    Alaska and Hawaii → "Pacific" (distinct climate, useful to separate)
    Non-matching states → null (edge case: non-US or territories)
    """
    return df.withColumn(
        "US_Region",
        F.when(F.col("State").isin(NE_STATES), "Northeast")
         .when(F.col("State").isin(SE_STATES), "Southeast")
         .when(F.col("State").isin(MW_STATES), "Midwest")
         .when(F.col("State").isin(SW_STATES), "Southwest")
         .when(F.col("State").isin(W_STATES),  "West")
         .when(F.col("State").isin("AK", "HI"),"Pacific")
         .otherwise(None)
    )


# ---------------------------------------------------------------------------
# Main transformation function
# ---------------------------------------------------------------------------

def transform_airport(spark: SparkSession) -> None:

    print("=" * 60)
    print("Silver | airports | Starting transformation")
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
    # 3. Rename columns to readable snake_case
    # ------------------------------------------------------------------
    df = df.withColumnRenamed("AIRPORT",  "Airport_Name") \
           .withColumnRenamed("CITY",     "City") \
           .withColumnRenamed("STATE",    "State") \
           .withColumnRenamed("COUNTRY",  "Country") \
           .withColumnRenamed("LATITUDE", "Latitude") \
           .withColumnRenamed("LONGITUDE","Longitude")
    # IATA_CODE kept as-is — it's the primary join key across all Silver tables

    # ------------------------------------------------------------------
    # 4. Trim and uppercase IATA_CODE for join safety
    # ------------------------------------------------------------------
    df = df.withColumn("IATA_CODE", F.upper(F.trim(F.col("IATA_CODE"))))

    # ------------------------------------------------------------------
    # 5. Trim all other string columns
    # ------------------------------------------------------------------
    string_cols = [
        field.name for field in df.schema.fields
        if str(field.dataType) == "StringType()" and field.name != "IATA_CODE"
    ]
    for col_name in string_cols:
        df = df.withColumn(col_name, F.trim(F.col(col_name)))

    # ------------------------------------------------------------------
    # 6. Round Latitude and Longitude to 5 decimal places
    #    Removes floating point noise from ingestion, keeps geo precision
    # ------------------------------------------------------------------
    df = df.withColumn("Latitude",  F.round(F.col("Latitude"),  5))
    df = df.withColumn("Longitude", F.round(F.col("Longitude"), 5))

    # ------------------------------------------------------------------
    # 7. Derive US_Region from State
    # ------------------------------------------------------------------
    df = add_us_region(df)

    # ------------------------------------------------------------------
    # 8. Derive Is_Alaska_Hawaii flag
    #    These airports have significantly different weather patterns.
    #    Flagging allows Gold layer to exclude or segment them easily.
    # ------------------------------------------------------------------
    df = df.withColumn(
        "Is_Alaska_Hawaii",
        F.when(F.col("State").isin("AK", "HI"), True)
         .otherwise(False)
    )

    # ------------------------------------------------------------------
    # 9. Silver_Load_Timestamp
    # ------------------------------------------------------------------
    df = df.withColumn("Silver_Load_Timestamp", F.current_timestamp())

    # ------------------------------------------------------------------
    # 10. Final column ordering
    # ------------------------------------------------------------------
    final_cols = [
        # Primary join key
        "IATA_CODE",

        # Location identifiers
        "Airport_Name", "City", "State", "Country",

        # Geography
        "Latitude", "Longitude",

        # Derived regional groupings
        "US_Region", "Is_Alaska_Hawaii",

        # Metadata
        "Silver_Load_Timestamp",
    ]

    df = df.select(*final_cols)

    # ------------------------------------------------------------------
    # 11. Sanity checks before write
    # ------------------------------------------------------------------
    silver_count = df.count()
    print(f"\n[Silver] Rows to write       : {silver_count:,}")
    print(f"[Silver] Output columns      : {len(df.columns)}")

    print(f"\n[Silver] US_Region distribution:")
    df.groupBy("US_Region").count().orderBy("US_Region").show()

    print(f"[Silver] Is_Alaska_Hawaii:")
    df.groupBy("Is_Alaska_Hawaii").count().show()

    print(f"[Silver] Null check — key columns:")
    df.select(
        F.count(F.when(F.col("IATA_CODE").isNull(),  1)).alias("IATA_Nulls"),
        F.count(F.when(F.col("State").isNull(),       1)).alias("State_Nulls"),
        F.count(F.when(F.col("US_Region").isNull(),   1)).alias("Region_Nulls"),
        F.count(F.when(F.col("Latitude").isNull(),    1)).alias("Lat_Nulls"),
        F.count(F.when(F.col("Longitude").isNull(),   1)).alias("Lon_Nulls"),
    ).show()
    df = df.toDF(*[c.lower() for c in df.columns])
    print(f"[Silver] Sample rows:")
    df.show(5, truncate=False)
    
    # ------------------------------------------------------------------
    # 12. Write Silver parquet — no partitioning (364 rows, static table)
    # ------------------------------------------------------------------
    df.write \
      .mode("overwrite") \
      .parquet(SILVER_PATH)

    print(f"\n[Silver] Written to          : {SILVER_PATH}")
    df.unpersist()
    print(f"[Silver] No partitioning     : static reference table (364 rows)")
    print("=" * 60)
    print("Silver | airports | Transformation complete ✓")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------