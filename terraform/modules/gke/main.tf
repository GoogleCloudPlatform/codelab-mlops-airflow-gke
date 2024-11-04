resource "google_container_cluster" "default" {
  name     = "gke-autopilot-basic"
  location = "us-central1"

  enable_autopilot = true

  # Set `deletion_protection` to `true` will ensure that one cannot
  # accidentally delete this instance by use of Terraform.
  deletion_protection = false
}
