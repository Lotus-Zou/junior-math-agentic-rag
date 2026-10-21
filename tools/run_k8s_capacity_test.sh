#!/usr/bin/env bash
set -euo pipefail

namespace="${NAMESPACE:-math-agent}"
job="${JOB_NAME:-math-agent-capacity-test}"
output_dir="${OUTPUT_DIR:-evaluation/reports/k8s-capacity-$(date +%Y%m%d%H%M%S)}"
capacity_overlay="${CAPACITY_OVERLAY:-deploy/overlays/capacity-test}"
mkdir -p "$output_dir"

python3 tools/verify_gpu_stack.py --output "$output_dir/gpu-preflight.json"
[[ -f "$capacity_overlay/kustomization.yaml" ]] || { echo "Missing $capacity_overlay: build and digest-pin app images first." >&2; exit 1; }
kubectl get --raw='/readyz' >"$output_dir/apiserver-readyz.txt"
kubectl get nodes -o wide >"$output_dir/nodes-before.txt"
kubectl get pods -n "$namespace" -o wide >"$output_dir/pods-before.txt"
kubectl top pods -n "$namespace" >"$output_dir/pod-top-before.txt" 2>&1 || true

kubectl delete job "$job" -n "$namespace" --ignore-not-found
kubectl apply -k "$capacity_overlay"
# Preserve logs even if an SLO gate fails and the Job exits nonzero.
test_status=0
while true; do
  state="$(kubectl get job "$job" -n "$namespace" -o json)"
  phase="$(python3 -c 'import json,sys; s=json.load(sys.stdin).get("status",{}); print("complete" if s.get("succeeded",0) else "failed" if s.get("failed",0) else "running")' <<< "$state")"
  if [[ "$phase" == complete ]]; then break; fi
  if [[ "$phase" == failed ]]; then test_status=1; break; fi
  sleep 10
done
kubectl logs "job/$job" -n "$namespace" >"$output_dir/load-test.log" 2>&1 || true
kubectl get pods -n "$namespace" -o wide >"$output_dir/pods-after.txt"
kubectl top pods -n "$namespace" >"$output_dir/pod-top-after.txt" 2>&1 || true
kubectl get hpa -n "$namespace" >"$output_dir/hpa-after.txt"
kubectl get events -n "$namespace" --sort-by=.lastTimestamp >"$output_dir/events.txt"

echo "Capacity evidence written to $output_dir"
exit "$test_status"
