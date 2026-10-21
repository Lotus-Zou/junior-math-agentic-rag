#!/usr/bin/env bash
# Install the real GPU/metrics stack without changing the current kubeconfig.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
: "${KUBECONFIG:?Set KUBECONFIG to the real cluster admin kubeconfig}"
profile="${GPU_PROFILE:-single-gpu}"
case "$profile" in
  single-gpu) manifest_dir=deploy/overlays/single-gpu ;;
  multi-gpu) manifest_dir=deploy/k8s ;;
  *) echo 'GPU_PROFILE must be single-gpu or multi-gpu.' >&2; exit 1 ;;
esac
image_overlay="${IMAGE_OVERLAY:-deploy/overlays/local-build}"
[[ -f "$image_overlay/kustomization.yaml" ]] || { echo "Missing $image_overlay: build and digest-pin app images first." >&2; exit 1; }
python3 tools/gpu_cluster_preflight.py --before-device-plugin --min-free-gib "${MIN_FREE_GIB:-50}"
# A failed scan prevents deployment. Reports preserve actual scanner findings.
bash tools/scan_gpu_images.sh
kubectl apply -f deploy/k8s/namespace.yaml
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
kubectl get secret math-agent-secrets -n math-agent >/dev/null
kubectl get secret math-agent-secrets -n math-agent -o json | python3 -c '
import base64,json,sys
d=json.load(sys.stdin).get("data",{})
key=base64.b64decode(d.get("OPENAI_API_KEY","")).decode()
if not key or key == "replace-me":
    raise SystemExit("A non-placeholder OPENAI_API_KEY is required for model authentication")
'

work_dir="$(mktemp -d)"
trap 'rm -rf -- "$work_dir"' EXIT
umask 077
openssl req -x509 -newkey rsa:3072 -nodes -days 90 \
  -subj /CN=math-agent-metrics-ca -addext basicConstraints=critical,CA:TRUE \
  -keyout "$work_dir/ca.key" -out "$work_dir/ca.crt" 2>/dev/null
issue_cert() {
  local namespace="$1" service="$2" secret="$3"
  openssl req -new -newkey rsa:2048 -nodes -subj "/CN=$service.$namespace.svc" \
    -addext "subjectAltName=DNS:$service,DNS:$service.$namespace,DNS:$service.$namespace.svc,DNS:$service.$namespace.svc.cluster.local" \
    -keyout "$work_dir/$service.key" -out "$work_dir/$service.csr" 2>/dev/null
  openssl x509 -req -days 90 -in "$work_dir/$service.csr" \
    -CA "$work_dir/ca.crt" -CAkey "$work_dir/ca.key" -CAcreateserial \
    -copy_extensions copy -out "$work_dir/$service.crt" 2>/dev/null
  kubectl create secret tls "$secret" -n "$namespace" \
    --key="$work_dir/$service.key" --cert="$work_dir/$service.crt" \
    --dry-run=client -o yaml | kubectl apply -f -
}
issue_cert kube-system metrics-server metrics-server-tls
issue_cert monitoring math-agent-prometheus-adapter math-agent-prometheus-adapter-tls
kubectl apply -k "$image_overlay"
ca_bundle="$(openssl base64 -A -in "$work_dir/ca.crt")"
for api in v1beta1.metrics.k8s.io v1beta1.custom.metrics.k8s.io; do
  kubectl patch apiservice "$api" --type=merge --patch "{\"spec\":{\"caBundle\":\"$ca_bundle\"}}"
done
# Reload serving certificates after renewal, including repeat installations.
kubectl rollout restart deployment/metrics-server -n kube-system
kubectl rollout restart deployment/math-agent-prometheus-adapter -n monitoring
kubectl rollout status daemonset/nvidia-device-plugin -n kube-system --timeout=180s
python3 tools/gpu_cluster_preflight.py
kubectl rollout status deployment/math-agent-model -n math-agent --timeout=30m
kubectl rollout status deployment/metrics-server -n kube-system --timeout=5m
kubectl rollout status deployment/math-agent-prometheus -n monitoring --timeout=5m
kubectl rollout status deployment/math-agent-prometheus-adapter -n monitoring --timeout=5m
kubectl wait --for=condition=Available apiservice/v1beta1.metrics.k8s.io --timeout=180s
kubectl wait --for=condition=Available apiservice/v1beta1.custom.metrics.k8s.io --timeout=180s
kubectl top nodes
echo 'TLS certificates expire in 90 days; renew before expiry by rerunning this script.'
echo 'Run tools/verify_gpu_stack.py before the capacity acceptance job.'
