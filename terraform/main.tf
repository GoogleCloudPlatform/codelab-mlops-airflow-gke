module "gcs" {
  source = "./modules/gcs"

  project_id     = var.project_id
  project_number = var.project_number
  region         = var.region
}