import os
from datetime import datetime
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

with DAG(
    dag_id="mlops_automated_pipeline",
    start_date=datetime(2025, 12, 31),
    schedule_interval="0 0 * * *",
    catchup=False,
) as dag:

    train = DockerOperator(
        task_id="model_training",
        image="gobong/mlops-trainer:latest",
        command="python -m src.main train",
        network_mode="bridge",
        auto_remove=True,

        mount_tmp_dir=False,

        environment={
            "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY"),
            "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2"),
            "S3_BUCKET_NAME": os.environ.get("S3_BUCKET_NAME"),
        },
    )

