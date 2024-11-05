import os
from datetime import datetime

from airflow import DAG
from airflow.models import Variable
from airflow.operators.dummy import DummyOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator

from kubernetes.client import models as k8s_models

GCP_PROJECT_ID = Variable.get("GCP_PROJECT_ID")
BUCKET_DATA_NAME = Variable.get("BUCKET_DATA_NAME")
HF_TOKEN = Variable.get("HF_TOKEN")
JOB_NAMESPACE = Variable.get("JOB_NAMESPACE", default_var="airflow")

with DAG(dag_id="mlops-dag",
            start_date=datetime(2024,8,1),
            schedule_interval="@hourly",
            catchup=False) as dag:
        
        # Step 1: Create GCS Folder
        create_folder = DummyOperator(
            task_id="create_folder_task"
        )

        # Step 2: Fetch raw data to GCS Bucket
        dataset_download = DummyOperator(
            task_id="dataset_download_task"
        )

        # Step 3: Run GKEJob for data preparation
        data_preparation = KubernetesPodOperator(
            task_id="data_pipeline_task",
            namespace=JOB_NAMESPACE,
            image="us-central1-docker.pkg.dev/mlops-codelab-440409/mlops-airflow-repo/data-pipeline@sha256:2635b79afb86d16324adeb9f27a9824dec78866edcba4c06f9dc1cb6a54cdaa0",
            name="data-preparation",
            service_account_name="airflow-mlops-sa",
            env_vars={
                    "PROJECT_ID":GCP_PROJECT_ID,
                    "BUCKET_DATA_NAME":BUCKET_DATA_NAME,
                    "DATASET_LIMIT": "1000",
                    "HF_TOKEN":HF_TOKEN
            }
        )

        # Step 4: Run GKEJob for fine tuning
        fine_tuning = KubernetesPodOperator(
            task_id="fine_tuning_task",
            namespace=JOB_NAMESPACE,
            image="us-central1-docker.pkg.dev/mlops-codelab-440409/mlops-airflow-repo/finetuning@sha256:48c56aa75cbae11c8ee57a34973edf2bc4364a6fe66cbe1887cb34f98fc382c2",
            name="fine-tuning",
            service_account_name="airflow-mlops-sa",
            startup_timeout_seconds=600,
            container_resources=k8s_models.V1ResourceRequirements(
                    requests={"cpu": "1", "memory": "8Gi", "nvidia.com/gpu": "1"},
                    limits={"cpu": "2", "memory": "16Gi", "nvidia.com/gpu": "1"}
            ),
            env_vars={
                    "BUCKET_DATA_NAME":BUCKET_DATA_NAME,
                    "HF_TOKEN":HF_TOKEN
            }
        )

        # Step 5: Run GKEJob for model serving
        model_evaluation = DummyOperator(
            task_id="model_evaluation_task"
        )

        # Step 6: Run GKEJob for model serving
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
                    "GCS_BUCKET":BUCKET_DATA_NAME,
                    "PREPARED_DATA_URL":"gs://mlops-airflow-model-489c/prepared_data.jsonl",
                    "HF_TOKEN":HF_TOKEN
            }
        )

        create_folder >> dataset_download >> data_preparation >> fine_tuning >> model_evaluation >> model_serving
