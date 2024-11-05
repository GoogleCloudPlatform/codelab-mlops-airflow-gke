# Building MLOps Workflows with Airflow 2 on GKE

## Data Ingestion & Preparation

### Enable GKE API service
```bash
gcloud services enable container
```

### Create a GKE autopilot
```bash
gcloud container clusters create-auto mlops-airflow --zone us-central1-a
```

### Generate docker images for DAG pipeline
```bash
cd services/data-pipeline
gcloud builds submit --tag us-central1-docker.pkg.dev/mlops-codelab/mlops-codelab-repo/data-pipeline
```

```bash
cd services/dataset-download
gcloud builds submit --tag us-central1-docker.pkg.dev/mlops-codelab/mlops-codelab-repo/dataset-download
```

```bash
cd services/finetuning
gcloud builds submit --tag us-central1-docker.pkg.dev/mlops-codelab/mlops-codelab-repo/finetuning
```