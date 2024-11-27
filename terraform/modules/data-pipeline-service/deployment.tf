locals {
  sa_name      = "data-pipeline-service-sa"
  service_name = "data-pipeline-service"
}

resource "google_service_account" "this" {
  project      = var.project_id
  account_id   = local.sa_name
  display_name = "Terraform-managed service account for data pipeline service"

}

resource "kubectl_manifest" "sa" {
  yaml_body = <<YAML
apiVersion: v1
kind: ServiceAccount
metadata:
  name: "${local.sa_name}"
  namespace: "${var.ns_name}"
  annotations:
    iam.gke.io/gcp-service-account: "${local.sa_name}@${var.project_id}.iam.gserviceaccount.com"
YAML
}

resource "google_service_account_iam_member" "this" {
  service_account_id = google_service_account.this.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${var.ns_name}/${local.sa_name}]"

  depends_on = [kubectl_manifest.sa]
}

# Grant the Service Account Access to GCS Bucket
resource "google_storage_bucket_iam_member" "bucket_access" {
  bucket = "finetuning-data-bucket" # Replace with your actual bucket name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${local.sa_name}@${var.project_id}.iam.gserviceaccount.com"
}

# Create a Kubernetes Job for data pipeline 
resource "kubectl_manifest" "this" {
  yaml_body = <<YAML
apiVersion: batch/v1
kind: Job
metadata:
  name: "${local.service_name}-job"
  namespace: ${var.ns_name}
spec:
  template:
    metadata:
      labels:
        app: "${local.service_name}"
    spec:
      serviceAccountName: ${local.sa_name}
      containers:
      - name: "${local.service_name}"
        image: ${var.region}-docker.pkg.dev/${var.project_id}/${var.artifactory_repo_name}/${local.service_name}:latest
        imagePullPolicy: Always
        resources:
          requests:
            cpu: "1"
            memory: "8Gi"  # Adjust if needed
          limits:
            cpu: "2"
            memory: "16Gi"  # Adjust if needed
        ports:
        - name: server-port
          containerPort: 8080
        env:
        # HuggingFace access token as k8s secret
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-secret
              key: HUGGING_FACE_TOKEN
      restartPolicy: Never
  backoffLimit: 3  # Number of retries if the job fails
YAML

  depends_on = [
    google_service_account_iam_member.this
  ]
}
