"""
spark_jobs/runner.py
----------------------
Thin job dispatcher. Called by every Airflow BashOperator via spark-submit.

This file is the ONLY entry point for the Airflow DAG.
It does a dynamic import of the right module and calls the right function,
so none of the existing bronze/silver/gold code needs to change.

Usage
-----
    spark-submit spark_jobs/runner.py --job <job_name>

Available job names
-------------------
    # Bronze
    bronze_flights
    bronze_weather
    bronze_airports
    bronze_cancelled

    # Silver
    silver_flights
    silver_weather
    silver_airports
    silver_cancelled

    # Gold
    gold_flight_performance
    gold_weather_delay_impact
    gold_cancellation_analysis
    gold_weather_cancellation
    gold_ops_summary_3month

Exit codes
----------
    0 — job completed successfully
    1 — job name not recognised (argparse error)
    2 — job raised an exception (logged, then re-raised so Airflow marks FAILED)
"""

from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from datetime import datetime

from pyspark.sql import SparkSession


# ---------------------------------------------------------------------------
# Job registry
# Maps job_name → (python_module_path, function_name)
# The function signature is always: func(spark: SparkSession) -> None
# ---------------------------------------------------------------------------
JOB_REGISTRY: dict[str, tuple[str, str]] = {
    # ── Bronze ──────────────────────────────────────────────────────────────
    "bronze_flights":   ("spark_jobs.bronze.flights_ingestion",   "ingest_flights"),
    "bronze_weather":   ("spark_jobs.bronze.weather_ingestion",   "ingest_weather"),
    "bronze_airports":  ("spark_jobs.bronze.airports_ingestion",  "ingest_airports"),
    "bronze_cancelled": ("spark_jobs.bronze.cancelled_ingestion", "ingest_cancelled"),
    # ── Silver ──────────────────────────────────────────────────────────────
    "silver_flights":   ("spark_jobs.silver.flights",   "transform_flights_silver"),
    "silver_weather":   ("spark_jobs.silver.weather",   "transform_weather"),
    "silver_airports":  ("spark_jobs.silver.airport",   "transform_airport"),
    "silver_cancelled": ("spark_jobs.silver.cancelled", "transform_cancelled"),
    # ── Gold ────────────────────────────────────────────────────────────────
    "gold_flight_performance":    ("spark_jobs.gold.flight_performance",    "build_flight_performance"),
    "gold_weather_delay_impact":  ("spark_jobs.gold.weather_delay_impact",  "build_weather_delay_impact"),
    "gold_cancellation_analysis": ("spark_jobs.gold.cancellation_analysis", "build_cancellation_analysis"),
    "gold_weather_cancellation":  ("spark_jobs.gold.weather_cancellation",  "build_weather_cancellation"),
    "gold_ops_summary_3month":    ("spark_jobs.gold.ops_summary_3month",    "build_ops_summary"),
}


# ---------------------------------------------------------------------------
# SparkSession factory
# Identical S3 + shuffle config to original main.py.
# app_name is set per-job so Spark UI shows which job is running.
# ---------------------------------------------------------------------------
def build_spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "2")
        # .config(
        #     "spark.jars.packages",
        #     "org.apache.hadoop:hadoop-aws:3.3.4",
        # )
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
        )
        .config(
            "spark.hadoop.fs.s3a.impl",
            "org.apache.hadoop.fs.s3a.S3AFileSystem",
        )
        .getOrCreate()
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _banner(text: str, width: int = 64) -> None:
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flight ETL job runner — called by Airflow BashOperator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available jobs:\n  " + "\n  ".join(sorted(JOB_REGISTRY)),
    )
    parser.add_argument(
        "--job",
        required=True,
        choices=sorted(JOB_REGISTRY.keys()),
        metavar="JOB_NAME",
        help=f"Job to execute. One of: {', '.join(sorted(JOB_REGISTRY))}",
    )
    args = parser.parse_args()
    job_name = args.job

    # ── Lookup ───────────────────────────────────────────────────────────────
    module_path, func_name = JOB_REGISTRY[job_name]

    _banner(f"FlightETL | {job_name} | start  [{_timestamp()}]")

    # ── Import function dynamically ──────────────────────────────────────────
    try:
        module   = importlib.import_module(module_path)
        job_func = getattr(module, func_name)
    except (ModuleNotFoundError, AttributeError) as exc:
        print(f"\n[FATAL] Could not import {func_name} from {module_path}")
        print(f"[FATAL] {exc}")
        sys.exit(2)

    # ── Build SparkSession ───────────────────────────────────────────────────
    spark = build_spark(f"FlightETL_{job_name}")
    spark.sparkContext.setLogLevel("WARN")

    # ── Run ──────────────────────────────────────────────────────────────────
    start_ts = datetime.now()
    try:
        job_func(spark)
        elapsed = (datetime.now() - start_ts).total_seconds()
        _banner(
            f"FlightETL | {job_name} | COMPLETE ✓  "
            f"[{_timestamp()}]  elapsed {elapsed:.1f}s"
        )

    except Exception:
        elapsed = (datetime.now() - start_ts).total_seconds()
        _banner(f"FlightETL | {job_name} | FAILED ✗  [{_timestamp()}]  elapsed {elapsed:.1f}s")
        traceback.print_exc()
        # Re-raise so spark-submit exits with code != 0
        # → Airflow BashOperator sees non-zero exit → marks task FAILED → triggers retry
        raise

    finally:
        spark.stop()


if __name__ == "__main__":
    main()