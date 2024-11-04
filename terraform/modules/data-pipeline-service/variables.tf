variable "project_id" {
  description = "The project ID to host the cluster in"
  type        = string
}

variable "artifactory_repo_name" {
  description = "The artifact registry name where image is stored"
  type        = string
}

variable "ns_name" {
  description = "The namespace name"
  type        = string
}

variable "region" {
  description = "The region to host the cluster in"
  type        = string
}

variable "hf_token" {
  description = "HuggingFace Access Token in base64 encoded form"
}
