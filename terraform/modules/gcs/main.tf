resource "random_id" "bucket_prefix" {
  byte_length = 8
}

locals {
  common_name = "${random_id.bucket_prefix.hex}-standard-bucket"

  bucket_name    = local.common_name
  project_number = var.project_number

#  zones = ["${var.region}-a", "${var.region}-b"]
}

data "google_client_config" "default" {}

# Get Project Number using a Data Source
data "google_project" "project" {
  project_id = var.project_id
}

resource "google_storage_bucket" "static" {
  name                        = local.bucket_name
  location                    = var.region
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  force_destroy               = true
  public_access_prevention    = "enforced"

  lifecycle {
    ignore_changes = all
  }
}