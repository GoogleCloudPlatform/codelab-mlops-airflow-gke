locals {
  ns_name               = "mlops"
  artifactory_repo_name = "llm-finetuning"
}

module "gcs" {
  source = "./modules/gcs"

  project_id     = var.project_id
  project_number = var.project_number
  region         = var.region
}

module "gke" {
  source = "./modules/gke"

  project_id = var.project_id
  region     = var.region
}

module "data-pipeline" {
  source = "./modules/data-pipeline-service"

  hf_token              = base64encode(var.hf_token)
  project_id            = var.project_id
  region                = var.region
  ns_name               = local.ns_name
  artifactory_repo_name = local.artifactory_repo_name

  depends_on = [module.gke]
}

module "finetuning" {
  source = "./modules/finetuning-service"

  hf_token              = base64encode(var.hf_token)
  ns_name               = local.ns_name
  project_id            = var.project_id
  region                = var.region
  artifactory_repo_name = local.artifactory_repo_name

  depends_on = [module.gke, module.data-pipeline]
} 
