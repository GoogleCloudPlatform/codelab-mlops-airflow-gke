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
KAGGLE_USERNAME = Variable.get("KAGGLE_USERNAME")
KAGGLE_KEY = Variable.get("KAGGLE_KEY")
JOB_NAMESPACE = Variable.get("JOB_NAMESPACE", default_var="airflow")

with DAG(dag_id="mlops-dag",
            catchup=False) as dag:

        # Step 1: Fetch raw data to GCS Bucket
        dataset_download = KubernetesPodOperator(
            task_id="dataset_download_task",
            namespace=JOB_NAMESPACE,
            image="us-central1-docker.pkg.dev/{{ var.value.GCP_PROJECT_ID }}/mlops-airflow-repo/dataset-download:latest",
            name="dataset-download",
            service_account_name="airflow-mlops-sa",
            env_vars={
                    "KAGGLE_USERNAME":KAGGLE_USERNAME,
                    "KAGGLE_KEY":KAGGLE_KEY,
                    "BUCKET_DATA_NAME":BUCKET_DATA_NAME
            }
        )

        # Step 2: Run GKEJob for data preparation
        data_preparation = KubernetesPodOperator(
            task_id="data_pipeline_task",
            namespace=JOB_NAMESPACE,
            image="us-central1-docker.pkg.dev/{{ var.value.GCP_PROJECT_ID }}/mlops-airflow-repo/data-pipeline:latest",
            name="data-preparation",
            service_account_name="airflow-mlops-sa",
            env_vars={
                    "PROJECT_ID":GCP_PROJECT_ID,
                    "BUCKET_DATA_NAME":BUCKET_DATA_NAME,
                    "DATASET_LIMIT": "1000",
                    "HF_TOKEN":HF_TOKEN
            }
        )

        # Step 3: Run GKEJob for fine tuning
        fine_tuning = KubernetesPodOperator(
            task_id="fine_tuning_task",
            namespace=JOB_NAMESPACE,
            image="us-central1-docker.pkg.dev/{{ var.value.GCP_PROJECT_ID }}/mlops-airflow-repo/finetuning:latest",
            name="fine-tuning",
            service_account_name="airflow-mlops-sa",
            startup_timeout_seconds=600,
            container_resources=k8s_models.V1ResourceRequirements(
                    requests={"nvidia.com/gpu": "1"},
                    limits={"nvidia.com/gpu": "1"}
            ),
            env_vars={
                    "BUCKET_DATA_NAME":BUCKET_DATA_NAME,
                    "HF_TOKEN":HF_TOKEN
            }
        )

        # Step 4: Run GKEJob for model evaluation
        model_evaluation = DummyOperator(
            task_id="model_evaluation_task"
        )

        # Step 5: Run GKEJob for model serving
        model_serving = KubernetesPodOperator(
            task_id="model_serving",
            namespace=JOB_NAMESPACE,
            image="us-central1-docker.pkg.dev/{{ var.value.GCP_PROJECT_ID }}/mlops-airflow-repo/inference:latest",
            name="model-serving",
            service_account_name="airflow-mlops-sa",
            container_resources=k8s_models.V1ResourceRequirements(
                    requests={"nvidia.com/gpu": "2"},
                    limits={"nvidia.com/gpu": "2"}
            ),
            env_vars={
                    "BUCKET_DATA_NAME":BUCKET_DATA_NAME,
                    "PREPARED_DATA_URL":"gs://mlops-airflow-model-489c/prepared_data.jsonl",
            }
        )

        dataset_download >> data_preparation >> fine_tuning >> model_evaluation >> model_serving
