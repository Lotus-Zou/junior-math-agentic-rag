#!/usr/bin/env python3
"""Validate manifests against the API without claiming that KWOK runs containers."""

import json
from pathlib import Path
import subprocess
import sys
import uuid

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.reproduce_experiments import write
from tools.gpu_cluster_preflight import inspect_nodes


def main():
    profiles = []
    for profile in ("deploy/k8s", "deploy/overlays/single-gpu"):
        rendered = subprocess.check_output(["kubectl", "kustomize", str(ROOT / profile)], text=True)
        resources = list(yaml.safe_load_all(rendered))
        prefix = "math-validation-"+uuid.uuid4().hex[:8]
        namespaces = {"math-agent":prefix, "monitoring":prefix+"-monitoring"}

        def remap(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "namespace" and item in namespaces:
                        value[key] = namespaces[item]
                    else:
                        remap(item)
            elif isinstance(value, list):
                for item in value:
                    remap(item)

        created = []
        try:
            for namespace in namespaces.values():
                subprocess.run(["kubectl", "create", "namespace", namespace], check=True, capture_output=True)
                created.append(namespace)
            for resource in resources:
                remap(resource)
            content = yaml.safe_dump_all([resource for resource in resources if resource["kind"] != "Namespace"])
            validation = subprocess.run(["kubectl", "apply", "--dry-run=server", "-f", "-"], input=content, text=True, capture_output=True)
        finally:
            for namespace in created:
                subprocess.run(["kubectl", "delete", "namespace", namespace, "--wait=false"], check=True, capture_output=True)
        profiles.append({"profile": profile,"resources":len(resources),"server_dry_run_passed":validation.returncode==0,
                         "stdout":validation.stdout,"stderr":validation.stderr})
    nodes = json.loads(subprocess.check_output(["kubectl","get","nodes","-o","json"], text=True))["items"]
    preflight = inspect_nodes(nodes)
    write("deployment", {"summary": {"manifests_valid":all(item["server_dry_run_passed"] for item in profiles),
                                      "real_gpu_kubernetes_available":preflight["passed"]},
                          "profiles":profiles,"gpu_preflight":preflight,
                          "scope":"Server-side Kubernetes validation in fresh temporary namespaces (removed afterwards), not an in-place storage migration test. The current node uses KWOK; no container startup, GPU Pod, probe execution, network forwarding or HPA metric pipeline is proven by this test."})


if __name__ == "__main__":
    main()
