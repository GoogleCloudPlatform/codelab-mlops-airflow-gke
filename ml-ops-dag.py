import os
from datetime import datetime

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator

from kubernetes.client import models as k8s_models

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "mlops-codelab-440409")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1-a")
CLUSTER_NAME = os.environ.get("CLUSTER_NAME", "mlops-airflow")
JOB_NAMESPACE = os.environ.get("JOB_NAMESPACE", "airflow")

with DAG(dag_id="mlops-dag",
            start_date=datetime(2024,8,1),
            schedule_interval="@hourly",
            catchup=False) as dag:
        
        # Step 1: Create GCS Folder
        create_folder = DummyOperator(
            task_id="create_folder_task"
        )

        # Step 2: Fetch raw data to GCS Bucket
        dataset_download = KubernetesPodOperator(
            task_id="dataset_download_task",
            namespace=JOB_NAMESPACE,
            image="us-central1-docker.pkg.dev/mlops-codelab-440409/mlops-airflow-repo/dataset-download@sha256:95fa8b282da5dd1ff035562dce8775b1df9a8547283022c53f94e3b9f4092a92",
            name="dataset-download",
            service_account_name="airflow-mlops-sa",
            env_vars={
                    "KAGGLE_USERNAME":"laurentgrangeau",
                    "KAGGLE_KEY":"c38a65c9f6e37ea0c29f07f078a24764",
                    "GCS_BUCKET":"mlops-airflow-model-489c"
            }
        )

        # Step 3: Run GKEJob for data preparation
        data_preparation = KubernetesPodOperator(
            task_id="data_pipeline_task",
            namespace=JOB_NAMESPACE,
            image="us-central1-docker.pkg.dev/mlops-codelab-440409/mlops-airflow-repo/data-pipeline@sha256:04bbf45002a6578bca8301983f8f2c99e647b50fbb3015350481fed4f70740fc",
            name="data-preparation",
            service_account_name="airflow-mlops-sa",
            env_vars={
                    "BUCKET_DATA_URL":"gs://mlops-airflow-model-489c/rotten_tomatoes_movie_reviews.csv",
                    "PREPARED_DATA_URL":"gs://mlops-airflow-model-489c/prepared_data.jsonl",
                    "HF_TOKEN":"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",
                    "PROJECT_ID":GCP_PROJECT_ID
            }
        )

        # Step 4: Run GKEJob for fine tuning
        fine_tuning = KubernetesPodOperator(
            task_id="fine_tuning_task",
            namespace=JOB_NAMESPACE,
            image="us-central1-docker.pkg.dev/mlops-codelab-440409/mlops-airflow-repo/finetuning@sha256:52a1564cfc286942c427f95d7c03afb79826658c4f64babb7ee1ad8ffba60eb8",
            name="fine-tuning",
            service_account_name="airflow-mlops-sa",
            startup_timeout_seconds=600,
            container_resources=k8s_models.V1ResourceRequirements(
                    requests={"cpu": "1", "memory": "8Gi", "nvidia.com/gpu": "1"},
                    limits={"cpu": "2", "memory": "16Gi", "nvidia.com/gpu": "1"}
            ),
            env_vars={
                    "GCS_BUCKET":"mlops-airflow-model-489c",
                    "PREPARED_DATA_URL":"gs://mlops-airflow-model-489c/prepared_data.jsonl",
                    "HF_TOKEN":"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"
            }
        )

        # Step 5: Run GKEJob for model serving
        model_serving = KubernetesPodOperator(
            task_id="model_serving",
            namespace=JOB_NAMESPACE,
            image="us-central1-docker.pkg.dev/mlops-codelab-440409/mlops-airflow-repo/finetuning@sha256:52a1564cfc286942c427f95d7c03afb79826658c4f64babb7ee1ad8ffba60eb8",
            name="fine-tuning",
            service_account_name="airflow-mlops-sa",
            startup_timeout_seconds=600,
            container_resources=k8s_models.V1ResourceRequirements(
                    requests={"cpu": "1", "memory": "8Gi", "nvidia.com/gpu": "1"},
                    limits={"cpu": "2", "memory": "16Gi", "nvidia.com/gpu": "1"}
            ),
            env_vars={
                    "GCS_BUCKET":"mlops-airflow-model-489c",
                    "PREPARED_DATA_URL":"gs://mlops-airflow-model-489c/prepared_data.jsonl",
                    "HF_TOKEN":"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"
            }
        )

        create_folder >> dataset_download >> data_preparation >> fine_tuning