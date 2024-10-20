# output "kubernetes_endpoint" {
#   description = "The cluster endpoint"
#   sensitive   = true
#   value       = google_container_cluster.primary.endpoint
# }

# output "cluster_ca_certificate" {
#   description = "The cluster ca certificate (base64 encoded)"
#   value       = google_container_cluster.primary.master_auth.0.cluster_ca_certificate
#   sensitive   = true
# }

# output "network_id" {
#   value       = module.gcp-network.network_id
#   description = "The ID of the VPC being created"
# }