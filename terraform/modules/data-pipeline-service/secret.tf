resource "kubectl_manifest" "secret" {
  yaml_body = <<YAML
    apiVersion: v1    
    kind: Secret
    metadata:
        name: hf-secret
        namespace: "${var.ns_name}"
    type: Opaque
    data:
        HUGGING_FACE_TOKEN: ${var.hf_token}
YAML

  depends_on = [
    kubectl_manifest.ns
  ]
}
