## These variables will be requested when running terraform. Please insert the appropriate information in CLI.
## Alternatively, you can uncomment the defaults here and replace with the appropriate information.
variable "project_id" {
  description = "The project ID to host the cluster in"
  type        = string
  default     = "mlops-airflow2"
}

variable "project_number" {
  description = "The project number to host the cluster in"
  type        = string
  default     = "888754845476"
}

variable "region" {
  description = "The region to host the cluster in"
  type        = string
  default     = "us-central1"
}

variable "hf_token" {
  description = "HuggingFace Access Token"
  type        = string
}
