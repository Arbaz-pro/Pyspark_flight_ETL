from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    "owner": "arbaz",
}

with DAG(
    dag_id="flight_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["spark", "etl"]
) as dag:

    # bronze_flights = BashOperator(
    #     task_id="bronze_flights",
    #     bash_command="""
    #     cd /mnt/d/Pyspark_ETL &&
    #     source ~/spark_env/venv/bin/activate &&
    #     python -m spark_jobs.main
    #     """
    # )
    bronze_flights = BashOperator(
    task_id="bronze_flights",
    bash_command="""
    cd /mnt/d/Pyspark_ETL &&
    /home/asus/spark_env/venv/bin/python -m spark_jobs.main
    """
)