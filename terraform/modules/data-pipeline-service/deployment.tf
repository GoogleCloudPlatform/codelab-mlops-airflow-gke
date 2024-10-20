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
  bucket = "finetuning-data-bucket"  # Replace with your actual bucket name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${local.sa_name}@${var.project_id}.iam.gserviceaccount.com"
}

resource "kubectl_manifest" "this" {
  yaml_body = <<YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: "${local.service_name}"
  namespace: ${var.ns_name}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: "${local.service_name}"
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
        ports:
        - name: server-port
          containerPort: 8080
YAML

  depends_on = [
    google_service_account_iam_member.this
  ]
}

resource "kubectl_manifest" "service" {
  yaml_body = <<YAML
apiVersion: v1
kind: Service
metadata:
  name: "${local.service_name}-svc"
  namespace: ${var.ns_name}
spec:
  type: ClusterIP
  selector:
    app: "${local.service_name}"
  ports:
    - protocol: TCP
      port: 8080
      targetPort: 8080

YAML

  depends_on = [kubectl_manifest.this]
}
