# MLOps Codelab with Airflow on GKE

## Set environment variable
```
export CODELAB_PREFIX=mlops-airflow
export PROJECT_NUMBER=$(gcloud projects list --filter="${DEVSHELL_PROJECT_ID}" --format="value(PROJECT_NUMBER)")

SUFFIX=$(echo $RANDOM | md5sum | head -c 4; echo;)
export CLUSTER_NAME=${CODELAB_PREFIX}
export CLUSTER_SA=sa-${CODELAB_PREFIX}
export BUCKET_LOGS_NAME=${CODELAB_PREFIX}-logs-${SUFFIX}
export BUCKET_DAGS_NAME=${CODELAB_PREFIX}-dags-${SUFFIX}
export BUCKET_DATA_NAME=${CODELAB_PREFIX}-data-${SUFFIX}
export REPO_NAME=${CODELAB_PREFIX}-repo
```

## Enable GKE API service
```
gcloud services enable container.googleapis.com
```

```
gcloud services enable cloudbuild.googleapis.com
```

```
gcloud services enable secretmanager.googleapis.com
```
## Create a VPC for the GKE cluster
```
gcloud compute networks create mlops --subnet-mode=auto
```

## Create the needed infrastructure (GKE, Bucket, Artifact Registry)
```
gcloud iam service-accounts create ${CLUSTER_SA} --display-name="SA for ${CLUSTER_NAME}"
```

```
gcloud projects add-iam-policy-binding ${DEVSHELL_PROJECT_ID} --member "serviceAccount:${CLUSTER_SA}@${DEVSHELL_PROJECT_ID}.iam.gserviceaccount.com" --role roles/container.defaultNodeServiceAccount
```

```
gcloud container clusters create ${CLUSTER_NAME} --zone us-central1-a --scopes=cloud-platform --gateway-api=standard --machine-type=g2-standard-4 --network=mlops --create-subnetwork name=mlops-subnet --enable-ip-alias --addons GcsFuseCsiDriver --workload-pool=${DEVSHELL_PROJECT_ID}.svc.id.goog --no-enable-insecure-kubelet-readonly-port --service-account=${CLUSTER_SA}@${DEVSHELL_PROJECT_ID}.iam.gserviceaccount.com
```

```
gcloud container clusters get-credentials ${CLUSTER_NAME} --location us-central1-a
```

```
gcloud storage buckets create gs://${BUCKET_LOGS_NAME} --location=us-central1
```

```
gcloud storage buckets create gs://${BUCKET_DAGS_NAME} --location=us-central1
```

```
gcloud storage buckets create gs://${BUCKET_DATA_NAME} --location=us-central1
```

```
gcloud artifacts repositories create ${REPO_NAME} --repository-format=docker --location=us-central1
```

```
gcloud artifacts repositories add-iam-policy-binding ${REPO_NAME} --member=serviceAccount:${CLUSTER_SA}@${DEVSHELL_PROJECT_ID}.iam.gserviceaccount.com --role=roles/artifactregistry.reader --location=us-central1
```

## Create the PV and PVC in GKE for Airflow DAGs storage
```
kubectl create namespace airflow
```

```
sed -i "s/BUCKET_DAGS_NAME/${BUCKET_DAGS_NAME}/g" manifests/pv-dags.yaml
```

```
sed -i "s/BUCKET_LOGS_NAME/${BUCKET_LOGS_NAME}/g" manifests/pv-logs.yaml
```

```
kubectl apply -f manifests/pv-dags.yaml
kubectl apply -f manifests/pv-logs.yaml
```

```
kubectl apply -f manifests/pvc-dags.yaml
kubectl apply -f manifests/pvc-logs.yaml
```

```
kubectl apply -f manifests/mlops-sa.yaml
```

## Add the necessary IAM permissions to access buckets from Airflow
```
gcloud projects add-iam-policy-binding ${DEVSHELL_PROJECT_ID} --member "principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${DEVSHELL_PROJECT_ID}.svc.id.goog/subject/ns/airflow/sa/airflow-scheduler" --role "roles/storage.objectUser"
```

```
gcloud projects add-iam-policy-binding ${DEVSHELL_PROJECT_ID} --member "principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${DEVSHELL_PROJECT_ID}.svc.id.goog/subject/ns/airflow/sa/airflow-triggerer" --role "roles/storage.objectUser"
```

```
gcloud projects add-iam-policy-binding ${DEVSHELL_PROJECT_ID} --member "principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${DEVSHELL_PROJECT_ID}.svc.id.goog/subject/ns/airflow/sa/airflow-worker" --role "roles/storage.objectUser"
```

```
gcloud projects add-iam-policy-binding ${DEVSHELL_PROJECT_ID} --member "principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${DEVSHELL_PROJECT_ID}.svc.id.goog/subject/ns/airflow/sa/airflow-worker" --role "roles/container.developer"
```

```
gcloud projects add-iam-policy-binding ${DEVSHELL_PROJECT_ID} --member "principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${DEVSHELL_PROJECT_ID}.svc.id.goog/subject/ns/airflow/sa/airflow-webserver" --role "roles/storage.objectUser"
```

```
gcloud projects add-iam-policy-binding ${DEVSHELL_PROJECT_ID} --member "principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${DEVSHELL_PROJECT_ID}.svc.id.goog/subject/ns/airflow/sa/airflow-mlops-sa" --role "roles/storage.objectUser"
```
## Install Airflow helm chart
```
helm repo add apache-airflow https://airflow.apache.org
helm repo update
```

```
helm upgrade --install airflow apache-airflow/airflow --namespace airflow -f values.yaml
```

## Connect to Airflow UI
```
kubectl -n airflow get svc/airflow-webserver --output jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

## Define a Google Cloud connection in Airflow UI
google_cloud_default

## Add IAM permissions to the Cloud Build service account
```
gcloud projects add-iam-policy-binding ${DEVSHELL_PROJECT_ID} --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" --role="roles/cloudbuild.serviceAgent"
```

```
gcloud projects add-iam-policy-binding ${DEVSHELL_PROJECT_ID} --member="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-cloudbuild.iam.gserviceaccount.com" --role="roles/logging.logWriter"
```

## Build the docker images
```
gcloud builds submit --tag us-central1-docker.pkg.dev/${DEVSHELL_PROJECT_ID}/${REPO_NAME}/data-pipeline
```

```
gcloud builds submit --tag us-central1-docker.pkg.dev/${DEVSHELL_PROJECT_ID}/${REPO_NAME}/finetuning
```

```
gcloud builds submit --tag us-central1-docker.pkg.dev/${DEVSHELL_PROJECT_ID}/${REPO_NAME}/inference
```

## Chage DAG variables
GCP_PROJECT_ID

## Upload your DAG file to the GCS bucket
```
gcloud storage cp ml-ops-dag.py gs://${BUCKET_DAGS_NAME}
```

Wait 30s at most

##

## BUG
Helm chart don't have a start job role predefined
Define a SA, Role and RoleBinding for job