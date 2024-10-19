provider "google" {
  project = var.project_id
  region  = var.region
}

/* provider "kubectl" {
  host                   = "https://${module.gke.kubernetes_endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(module.gke.cluster_ca_certificate)
} */

/* provider "google-beta" {
  project = var.project_id
  region  = var.region
} */

data "google_client_config" "default" {}
