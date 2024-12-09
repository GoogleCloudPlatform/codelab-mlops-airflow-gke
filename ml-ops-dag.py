# * Copyright 2022 Google LLC
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

import yaml

from os import path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator

from kubernetes import client, config
from kubernetes.client import models
from kubernetes.client.rest import ApiException

GCP_PROJECT_ID = Variable.get("GCP_PROJECT_ID")
BUCKET_DATA_NAME = Variable.get("BUCKET_DATA_NAME")
HF_TOKEN = Variable.get("HF_TOKEN")
KAGGLE_USERNAME = Variable.get("KAGGLE_USERNAME")
KAGGLE_KEY = Variable.get("KAGGLE_KEY")
JOB_NAMESPACE = Variable.get("JOB_NAMESPACE", default_var="airflow")

def model_serving():
    config.load_incluster_config()
    k8s_apps_v1 = client.AppsV1Api()
    k8s_core_v1 = client.CoreV1Api()

    while True:
        try:
            k8s_apps_v1.delete_namespaced_deployment(
                    namespace="airflow",
                    name="inference-deployment",
                    body=client.V1DeleteOptions(
                    propagation_policy="Foreground", grace_period_seconds=5
                    )
            )
        except ApiException:
            break
    print("Deployment inference-deployment deleted")
    
    with open(path.join(path.dirname(__file__), "inference.yaml")) as f:
        dep = yaml.safe_load(f)
        resp = k8s_apps_v1.create_namespaced_deployment(
            body=dep, namespace="airflow")
        print(f"Deployment created. Status='{resp.metadata.name}'")
    
    while True:
        try:
            k8s_core_v1.delete_namespaced_service(
                    namespace="airflow",
                    name="llm-service",
                    body=client.V1DeleteOptions(
                    propagation_policy="Foreground", grace_period_seconds=5
                    )
            )
        except ApiException:
            break
    print("Service llm-service deleted")

    with open(path.join(path.dirname(__file__), "inference-service.yaml")) as f:
        dep = yaml.safe_load(f)
        resp = k8s_core_v1.create_namespaced_service(
            body=dep, namespace="airflow")
        print(f"Service created. Status='{resp.metadata.name}'")

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
            image="us-central1-docker.pkg.dev/{{ var.value.GCP_PROJECT_ID }}/mlops-airflow-repo/data-preparation:latest",
            name="data-preparation",
            service_account_name="airflow-mlops-sa",
            env_vars={
                    "GCP_PROJECT_ID":GCP_PROJECT_ID,
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
            container_resources=models.V1ResourceRequirements(
                    requests={"nvidia.com/gpu": "1"},
                    limits={"nvidia.com/gpu": "1"}
            ),
            env_vars={
                    "BUCKET_DATA_NAME":BUCKET_DATA_NAME,
                    "HF_TOKEN":HF_TOKEN
            }
        )

        # Step 4: Run GKE Deployment for model serving
        model_serving = PythonOperator(
            task_id="model_serving",
            python_callable=model_serving
        )

        dataset_download >> data_preparation >> fine_tuning >> model_serving
