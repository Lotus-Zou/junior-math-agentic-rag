#!/usr/bin/env bash
# Build both application images and emit an immutable Kustomize overlay.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
builder="${CONTAINER_BUILDER:-}"
if [[ -z "$builder" ]]; then
  if command -v docker >/dev/null; then
    builder=docker
  elif command -v nerdctl >/dev/null; then
    builder=nerdctl
  else
    echo 'Install a container builder (nerdctl or docker) on the real GPU node.' >&2
    exit 1
  fi
fi
command -v "$builder" >/dev/null || { echo "Builder not found: $builder" >&2; exit 1; }
registry="${IMAGE_REGISTRY:-localhost:5000}"
tag="${IMAGE_TAG:-$(date -u +%Y%m%d%H%M%S)}"
profile="${GPU_PROFILE:-single-gpu}"
case "$profile" in
  single-gpu) resources=../single-gpu ;;
  multi-gpu) resources=../k8s ;;
  *) echo 'GPU_PROFILE must be single-gpu or multi-gpu.' >&2; exit 1 ;;
esac
ai_ref="$registry/math-agent-ai:$tag"
gateway_ref="$registry/math-agent-gateway:$tag"
"$builder" build --pull=false -t "$ai_ref" .
"$builder" build --pull=false -t "$gateway_ref" gateway
"$builder" push "$ai_ref"
"$builder" push "$gateway_ref"
ai_digest="$("$builder" image inspect "$ai_ref" --format '{{index .RepoDigests 0}}' 2>/dev/null || true)"
gateway_digest="$("$builder" image inspect "$gateway_ref" --format '{{index .RepoDigests 0}}' 2>/dev/null || true)"
if [[ "$ai_digest" != *@sha256:* || "$gateway_digest" != *@sha256:* ]]; then
  echo 'Builder did not return immutable RepoDigests. Push to a registry, then rerun with that registry.' >&2
  exit 1
fi
overlay="$repo_dir/deploy/overlays/local-build"
mkdir -p "$overlay"
printf '%s\n' \
  'apiVersion: kustomize.config.k8s.io/v1beta1' \
  'kind: Kustomization' \
  'resources:' \
  "  - $resources" \
  'images:' \
  '  - name: math-agent-ai' \
  "    newName: ${ai_digest%@*}" \
  "    digest: ${ai_digest##*@}" \
  '  - name: math-agent-gateway' \
  "    newName: ${gateway_digest%@*}" \
  "    digest: ${gateway_digest##*@}" \
  > "$overlay/kustomization.yaml"
echo "Wrote $overlay/kustomization.yaml"
