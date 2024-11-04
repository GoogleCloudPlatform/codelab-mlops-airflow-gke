# TBD: Do we need a self-created SA or should we use default SA?
resource "random_id" "sa_prefix" {
  byte_length = 8
}

locals {
  service_account_default_name = "tf-gcs-${random_id.sa_prefix.hex}"
}

resource "google_service_account" "cluster_service_account" {
  project      = var.project_id
  account_id   = local.service_account_default_name
  display_name = "Terraform-managed service account for GCS ${local.bucket_name}"
}
