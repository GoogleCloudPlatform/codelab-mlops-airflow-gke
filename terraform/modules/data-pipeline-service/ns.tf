resource "kubectl_manifest" "ns" {
  yaml_body = <<YAML
apiVersion: v1
kind: Namespace
metadata:
  name: "${var.ns_name}"
YAML
}
