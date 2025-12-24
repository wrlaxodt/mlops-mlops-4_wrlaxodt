from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
sys.path.append("/opt/airflow")

from src.data_prepare.main import run_data_prepare
from src.train.train import run_train
from src.inference.inference import load_checkpoint


# ------------------------
# Task functions
# ------------------------

def data_extraction():
    return run_data_prepare(
        start_page=1,
        end_page=1,
        output_csv="dataset/watch_log.csv",
    )


def data_validation():
    import os
    path = "/opt/airflow/dataset/watch_log.csv"
    if not os.path.exists(path):
        raise FileNotFoundError("watch_log.csv not found")

    import pandas as pd
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Dataset is empty")

    print(f"Data validation passed. rows={len(df)}")


def model_training():
    result = run_train(
        model_name="movie_predictor",
        num_epochs=10,
        batch_size=64,
    )
    print(result)
    return result


def model_validation():
    checkpoint = load_checkpoint(
        model_name="movie_predictor",
        model_ext="pkl",
    )
    print("Model validation success")


# ------------------------
# DAG definition
# ------------------------

with DAG(
    dag_id="mlops_automated_pipeline",
    start_date=datetime(2025, 12, 24),
    schedule_interval=None,
    catchup=False,
    tags=["mlops", "training"],
) as dag:

    t1 = PythonOperator(
        task_id="data_extraction",
        python_callable=data_extraction,
    )

    t2 = PythonOperator(
        task_id="data_validation",
        python_callable=data_validation,
    )

    t3 = PythonOperator(
        task_id="model_training",
        python_callable=model_training,
    )

    t4 = PythonOperator(
        task_id="model_validation",
        python_callable=model_validation,
    )

    t1 >> t2 >> t3 >> t4

