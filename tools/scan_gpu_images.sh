#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${IMAGE_SCAN_DIR:-$repo_dir/evaluation/reports/gpu-image-scans}"
command -v trivy >/dev/null || { echo 'Install Trivy before image vulnerability acceptance.' >&2; exit 1; }
mkdir -p "$output_dir"
image_list="$(python3 - "$repo_dir" <<'PY'
import json,sys
from pathlib import Path
paths = sorted((Path(sys.argv[1]) / 'deploy/k8s/artifacts').glob('*.json'))
paths = [p for p in paths if 'pinned' in json.loads(p.read_text())]
assert {p.stem for p in paths} == {'vllm','device-plugin','metrics-server','prometheus','prometheus-adapter','mysql','redis'}, 'Incomplete image locks'
for path in paths:
    data = json.loads(path.read_text())
    print(path.stem + '\t' + data['mirror_pinned'])
PY
)"
while IFS=$'\t' read -r name reference; do
  trivy image --image-src remote --timeout 30m --scanners vuln \
    --severity HIGH,CRITICAL --exit-code 1 --format json \
    --output "$output_dir/$name.json" "$reference"
done <<< "$image_list"
echo "All locked images passed HIGH/CRITICAL vulnerability checks: $output_dir"
