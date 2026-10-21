#!/usr/bin/env python3
"""Read-only host checks; never call KWOK an actual GPU deployment."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check(argv):
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=25)
        return {'command': argv, 'exit_code': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr[-3000:]}
    except (OSError, subprocess.SubprocessError) as exc:
        return {'command': argv, 'exit_code': None, 'error': str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from tools.gpu_cluster_preflight import inspect_nodes
    probes = [check(argv) for argv in [
        ['id'], ['sudo', '-n', 'true'], ['unshare', '-Ur', 'true'],
        ['rootlesskit', '--net=host', 'true'],
        ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
        ['kubectl', 'config', 'current-context'], ['kubectl', 'get', 'nodes', '-o', 'json']]]
    nodes = probes[-1]
    try:
        cluster = inspect_nodes(json.loads(nodes['stdout'])['items']) if nodes.get('exit_code') == 0 else {'passed': False, 'errors': ['No accessible real cluster']}
    except (ValueError, KeyError) as exc:
        cluster = {'passed': False, 'errors': ['Invalid node inventory: '+str(exc)]}
    report = {'generated_at': datetime.now(timezone.utc).isoformat(), 'checks': probes,
              'cluster_preflight': cluster, 'real_gpu_kubernetes_accepted': False,
              'runtime_binaries': {name: shutil.which(name) for name in ('docker', 'podman', 'containerd', 'nerdctl', 'k3s', 'nvidia-container-runtime')},
              'free_disk_gib': round(shutil.disk_usage('.').free/1024**3, 2),
              'policy': 'Host GPU presence is not a GPU Pod acceptance. Preflight and manifests alone do not establish running inference or capacity.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['real_gpu_kubernetes_accepted'] else 1)


if __name__ == '__main__':
    main()
