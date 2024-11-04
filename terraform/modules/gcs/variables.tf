variable "project_id" {
  description = "The project ID to host the cluster in"
  type        = string
}

variable "project_number" {
  description = "The project number to host the cluster in"
  type        = string
}

variable "region" {
  description = "The region to host the cluster in"
  type        = string
}

variable "bucket_name" {
  type        = string
  description = "GCS bucket name"
  default     = "gcs-bucket"
}
