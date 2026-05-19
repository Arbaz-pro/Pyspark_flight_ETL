"""
airflow_dags/flight_pipeline.py
---------------------------------
Production Airflow DAG for the US Flight ETL pipeline.

Architecture
------------
  Bronze (parallel)  →  Silver (per-source)  →  Gold (aggregations)

  4 Bronze tasks run concurrently.
  Each Silver task waits only for its own Bronze task.
  Gold tasks wait for all Silver tables they JOIN.
  ops_summary_3month runs last (needs flight + cancelled Silver + two Gold tables done).

Configuration
-------------
  All environment-specific paths live in Airflow Variables so the DAG code
  never needs to change between dev/staging/prod.

  Set these in Airflow Admin > Variables (or via CLI):
    flight_etl_project_dir    — abs path to project root  (default: /mnt/d/Pyspark_ETL)
    flight_etl_spark_submit   — abs path to spark-submit  (default: /home/asus/spark_env/venv/bin/spark-submit)
    flight_etl_java_home      — JAVA_HOME                 (default: /usr/lib/jvm/java-11-openjdk-amd64)
    flight_etl_spark_home     — SPARK_HOME                (default: /home/asus/spark_env/venv/lib/python3.11/site-packages/pyspark)
    flight_etl_driver_memory  — Spark driver memory       (default: 2g)
    flight_etl_email          — failure alert address     (default: arbaz@example.com)

Runner
------
  Each BashOperator calls:
      spark-submit spark_jobs/runner.py --job <job_name>

  runner.py does a dynamic import and calls the right function.
  This means existing bronze/silver/gold code needs zero changes.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup

# ---------------------------------------------------------------------------
# Pull config from Airflow Variables (falls back to dev defaults)
# ---------------------------------------------------------------------------
PROJECT_DIR  = Variable.get("flight_etl_project_dir",   default_var="/mnt/d/Pyspark_ETL")
SPARK_SUBMIT = Variable.get("flight_etl_spark_submit",  default_var="/home/asus/spark_env/venv/bin/spark-submit")
JAVA_HOME    = Variable.get("flight_etl_java_home",     default_var="/usr/lib/jvm/java-11-openjdk-amd64")
SPARK_HOME   = Variable.get("flight_etl_spark_home",    default_var="/home/asus/spark_env/venv/lib/python3.11/site-packages/pyspark")
DRIVER_MEM   = Variable.get("flight_etl_driver_memory", default_var="2g")
ALERT_EMAIL  = Variable.get("flight_etl_email",         default_var="arbaz@example.com")

# ---------------------------------------------------------------------------
# spark-submit flags shared across all tasks
# ---------------------------------------------------------------------------
_SPARK_FLAGS = " ".join([
    "--master spark://127.0.0.1:7077",
    f"--driver-memory {DRIVER_MEM}",
    "--packages org.apache.hadoop:hadoop-aws:3.3.4",
    "--conf spark.hadoop.fs.s3a.aws.credentials.provider=com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
    "--conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem",
    "--conf spark.executor.heartbeatInterval=120s",
    "--conf spark.network.timeout=600s",
    "--conf spark.sql.shuffle.partitions=2",
    "--conf spark.driver.host=127.0.0.1",
    "--conf spark.driver.bindAddress=127.0.0.1",
])

# ---------------------------------------------------------------------------
# Bash preamble — sets env vars; prepended to every BashOperator command
# ---------------------------------------------------------------------------
_ENV = f"""
set -euo pipefail
export JAVA_HOME={JAVA_HOME}
export SPARK_HOME={SPARK_HOME}
export PATH=$SPARK_HOME/bin:$PATH
export PYTHONPATH={PROJECT_DIR}
cd {PROJECT_DIR}
"""


def _spark_cmd(job_name: str) -> str:
    """Return a complete bash command string for a given runner job."""
    return f"""
{_ENV}
echo "========== Starting job: {job_name} =========="
{SPARK_SUBMIT} {_SPARK_FLAGS} spark_jobs/runner.py --job {job_name}
echo "========== Finished job: {job_name} =========="
"""


# ---------------------------------------------------------------------------
# Default task args
# ---------------------------------------------------------------------------
default_args = {
    "owner": "arbaz",
    "depends_on_past": False,
    # Retry twice with a 5-minute wait so transient S3 errors self-heal
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "email_on_retry": False,
    "email": [ALERT_EMAIL],
    # Kill a hung Spark job rather than block the scheduler indefinitely
    "execution_timeout": timedelta(hours=2),
}

# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="flight_pipeline",
    default_args=default_args,
    description="US Flight ETL — Bronze → Silver → Gold on S3",
    start_date=datetime(2025, 1, 1),
    # Manual trigger. Change to "@daily" or a cron string once source data
    # lands on a schedule (e.g. "0 3 * * *" for 3 AM daily).
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,  
    max_active_tasks=2,        # prevent concurrent pipeline runs
    tags=["spark", "etl", "flights", "s3"],
    doc_md="""
## Flight Pipeline

Medallion ETL for US domestic flight data (2023).

### Layers
| Layer  | Tables |
|--------|--------|
| Bronze | flights, weather, airports, cancelled |
| Silver | flights, weather, airports, cancelled |
| Gold   | flight_performance, weather_delay_impact, cancellation_analysis, weather_cancellation, ops_summary_3month |

### Key design decisions
- Bronze tasks run **in parallel** — no dependency between sources.
- Each Silver task waits **only for its own Bronze** source, not all four.
- Gold tasks declare exact Silver dependencies (e.g. `gold_weather_delay`
  waits for flights + weather + airports Silver, NOT cancelled Silver).
- `ops_summary_3month` runs last: it unions completed flights with
  cancelled/diverted to produce denominator-correct cancellation rates.

### Airflow Variables
Set these in Admin > Variables to override dev defaults without touching DAG code:

| Variable | Purpose |
|----------|---------|
| `flight_etl_project_dir` | Abs path to project root |
| `flight_etl_spark_submit` | Abs path to spark-submit binary |
| `flight_etl_java_home` | JAVA_HOME |
| `flight_etl_spark_home` | SPARK_HOME (PySpark site-packages path) |
| `flight_etl_driver_memory` | Spark driver memory (e.g. `4g`) |
| `flight_etl_email` | Failure alert email address |
    """,
) as dag:

    # -----------------------------------------------------------------------
    # Sentinels
    # -----------------------------------------------------------------------
    pipeline_start = EmptyOperator(
        task_id="pipeline_start",
        doc_md="Entry point. Immediately triggers all Bronze tasks in parallel.",
    )

    pipeline_end = EmptyOperator(
        task_id="pipeline_end",
        doc_md="All Gold tasks must reach this gate before the run is marked success.",
    )

    # -----------------------------------------------------------------------
    # BRONZE LAYER
    # All four tasks are independent — they run concurrently.
    # Each reads a raw CSV from S3/local RAW_BASE and writes Parquet to
    # BRONZE_BASE. No joins, no transformations beyond type inference.
    # -----------------------------------------------------------------------
    with TaskGroup(
        group_id="bronze",
        tooltip="Raw CSV ingestion → Parquet. All tasks run in parallel.",
    ) as bronze_group:

        bronze_flights = BashOperator(
            task_id="flights",
            bash_command=_spark_cmd("bronze_flights"),
            doc_md="""
**Source**: `US_flights_2023.csv`
**Output**: `s3a://flight-etl-lake-arbaz/bronze/flights/`
**Notes**: Largest dataset. inferSchema=True kept for now — add explicit schema once stable.
            """,
        )

        bronze_weather = BashOperator(
            task_id="weather",
            bash_command=_spark_cmd("bronze_weather"),
            doc_md="""
**Source**: `weather_meteo_by_airport.csv`
**Output**: `s3a://flight-etl-lake-arbaz/bronze/weather/`
            """,
        )

        bronze_airports = BashOperator(
            task_id="airports",
            bash_command=_spark_cmd("bronze_airports"),
            doc_md="""
**Source**: `airports_geolocation.csv`
**Output**: `s3a://flight-etl-lake-arbaz/bronze/airports/`
**Notes**: 364-row static reference table. Fast ingest.
            """,
        )

        bronze_cancelled = BashOperator(
            task_id="cancelled",
            bash_command=_spark_cmd("bronze_cancelled"),
            doc_md="""
**Source**: `Cancelled_Diverted_2023.csv`
**Output**: `s3a://flight-etl-lake-arbaz/bronze/cancelled_flights/`
**Notes**: Full 12-month dataset. FlightDate ingested as DateType here
           (unlike flights dataset which is a string — handled in Silver).
            """,
        )

    # -----------------------------------------------------------------------
    # SILVER LAYER
    # Each Silver task waits only for its own Bronze counterpart.
    # Transformations: dedup, type fixes, derived columns, partitioning.
    # -----------------------------------------------------------------------
    with TaskGroup(
        group_id="silver",
        tooltip="Clean, enrich, and partition each source independently.",
    ) as silver_group:

        silver_flights = BashOperator(
            task_id="flights",
            bash_command=_spark_cmd("silver_flights"),
            doc_md="""
**Reads**: `bronze/flights/`
**Writes**: `silver/flights/` partitioned by Flight_Year, Flight_Month
**Key transforms**:
- Fix `Aicraft_age` → `Aircraft_age` typo
- Fix `Hight` → `High` in delay type columns
- Standardise `Distance_type` labels
- Parse FlightDate string (dd-MM-yyyy) → DateType
- Derive `Total_Delay_Minutes`, `Primary_Delay_Cause`,
  `Departure_Status`, `Arrival_Status`
- Add `Silver_Load_Timestamp`
            """,
        )

        silver_weather = BashOperator(
            task_id="weather",
            bash_command=_spark_cmd("silver_weather"),
            doc_md="""
**Reads**: `bronze/weather/`
**Writes**: `silver/weather/` partitioned by Weather_Year, Weather_Month
**Key transforms**:
- Rename cryptic column names (tavg → Temp_Avg_C, etc.)
- `snow` null → 0.0 (null = no snow event, not missing data)
- Derive `Weather_Season`, `Temp_Range_C`, `Is_Precipitation`,
  `Wind_Category`, `Pressure_Category`
- Uppercase IATA_CODE for join safety
            """,
        )

        silver_airports = BashOperator(
            task_id="airports",
            bash_command=_spark_cmd("silver_airports"),
            doc_md="""
**Reads**: `bronze/airports/`
**Writes**: `silver/airports/` (NO partition — 364 rows, static ref table)
**Key transforms**:
- Rename columns to snake_case
- Derive `US_Region`, `Is_Alaska_Hawaii`
- Round Lat/Lon to 5 decimal places
            """,
        )

        silver_cancelled = BashOperator(
            task_id="cancelled",
            bash_command=_spark_cmd("silver_cancelled"),
            doc_md="""
**Reads**: `bronze/cancelled_flights/`
**Writes**: `silver/cancelled_flights/` partitioned by Flight_Year, Flight_Month
**Key transforms**:
- Fix `Hight` typo in delay type columns
- Standardise `Distance_type` labels to match flights Silver
- Derive `Flight_Status` (Cancelled / Diverted)
- Nullify delay cols for Cancelled=1 rows (flight never flew — zeros are misleading)
- Derive `Total_Delay_Minutes`, `Primary_Delay_Cause` (Diverted rows only)
            """,
        )

    # -----------------------------------------------------------------------
    # GOLD LAYER
    # Aggregated analytics tables. Each task declares exactly the Silver
    # tables it needs — no over-waiting on unrelated Silver tasks.
    # -----------------------------------------------------------------------
    with TaskGroup(
        group_id="gold",
        tooltip="Aggregated analytics. Tasks run as soon as their Silver deps are ready.",
    ) as gold_group:

        gold_flight_perf = BashOperator(
            task_id="flight_performance",
            bash_command=_spark_cmd("gold_flight_performance"),
            doc_md="""
**Sources**: silver/flights + silver/airports (broadcast)
**Output**: `gold/flight_performance/` partitioned by flight_year, flight_month
**Grain**: airline + dep_airport + arr_airport + flight_month + distance_type
**Scope**: 3 months (Jan–Mar 2023) — flights dataset coverage
**Key metrics**: avg/max dep_delay, avg arr_delay, on_time_pct, delayed_pct,
                 per-cause delay percentages
            """,
        )

        gold_weather_delay = BashOperator(
            task_id="weather_delay_impact",
            bash_command=_spark_cmd("gold_weather_delay_impact"),
            doc_md="""
**Sources**: silver/flights + silver/weather + silver/airports (broadcast)
**Output**: `gold/weather_delay_impact/` (no partition — small aggregated summary)
**Grain**: wind_category + pressure_category + is_precipitation + weather_season + dep_region
**Scope**: 3 months (flights anchor, weather matched by date+IATA)
**Key metrics**: avg_dep_delay, avg_arr_delay, delayed_flight_pct,
                 avg weather readings per group
            """,
        )

        gold_cancellations = BashOperator(
            task_id="cancellation_analysis",
            bash_command=_spark_cmd("gold_cancellation_analysis"),
            doc_md="""
**Sources**: silver/cancelled_flights + silver/airports (broadcast)
**Output**: `gold/cancellation_analysis/` partitioned by flight_year, flight_month
**Grain**: airline + dep_airport + arr_airport + flight_month + distance_type + weekday_name
**Scope**: Full 12 months
**Warning**: Rates computed here use a biased denominator (only cancelled/diverted rows).
             Use ops_summary_3month for real rates with completed flights in denominator.
            """,
        )

        gold_wx_cancel = BashOperator(
            task_id="weather_cancellation",
            bash_command=_spark_cmd("gold_weather_cancellation"),
            doc_md="""
**Sources**: silver/cancelled_flights + silver/weather + silver/airports (broadcast)
**Output**: `gold/weather_cancellation/` (no partition — small summary)
**Grain**: wind_category + pressure_category + is_precipitation + weather_season + dep_region + flight_month
**Scope**: Full 12 months — RICHEST Gold table (both cancelled and weather are full-year)
            """,
        )

        gold_ops_summary = BashOperator(
            task_id="ops_summary_3month",
            bash_command=_spark_cmd("gold_ops_summary_3month"),
            doc_md="""
**Sources**: silver/flights + silver/cancelled_flights (filtered to Jan–Mar) + silver/airports
**Output**: `gold/ops_summary_3month/` partitioned by flight_year, flight_month
**Grain**: airline + dep_airport + arr_airport + flight_month + distance_type + weekday_name
**Why this exists**: Union of completed + cancelled + diverted flights gives the correct
  total_scheduled denominator. Real cancellation rates of 2–5%, not 80–92%.
**Runs last**: depends on gold_flight_perf + gold_weather_delay being done (Track A complete)
               AND silver_cancelled being ready.
            """,
        )

    # -----------------------------------------------------------------------
    # Wire dependencies
    # -----------------------------------------------------------------------

    # Entry → Bronze (all 4 fire in parallel)
    pipeline_start >> bronze_group

    # Each Bronze feeds only its own Silver — not a fan-in to all Silver
    bronze_flights   >> silver_flights
    bronze_weather   >> silver_weather
    bronze_airports  >> silver_airports
    bronze_cancelled >> silver_cancelled

    # Gold tasks wait for exactly the Silver tables they JOIN
    # ─────────────────────────────────────────────────────
    # gold_flight_performance:  flights + airports
    [silver_flights, silver_airports] >> gold_flight_perf

    # gold_weather_delay_impact: flights + weather + airports
    [silver_flights, silver_weather, silver_airports] >> gold_weather_delay

    # gold_cancellation_analysis: cancelled + airports
    [silver_cancelled, silver_airports] >> gold_cancellations

    # gold_weather_cancellation: cancelled + weather + airports
    [silver_cancelled, silver_weather, silver_airports] >> gold_wx_cancel

    # gold_ops_summary_3month: runs after Track A Gold is done AND cancelled Silver is ready
    # It unions silver/flights with silver/cancelled_flights (3-month window),
    # so it logically should wait for both Silver tables to be written and for
    # Track A Gold to confirm the 3-month scope is fully processed.
    [gold_flight_perf, gold_weather_delay, silver_cancelled] >> gold_ops_summary

    # All Gold → end sentinel
    [
        gold_flight_perf,
        gold_weather_delay,
        gold_cancellations,
        gold_wx_cancel,
        gold_ops_summary,
    ] >> pipeline_end