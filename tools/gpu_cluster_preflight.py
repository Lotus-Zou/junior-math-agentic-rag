#!/usr/bin/env python3
"""Reject simulated nodes before a GPU deployment or capacity test."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def inspect_nodes(nodes: list[dict], require_gpu: bool = True) -> dict:
    errors, inventory = [], []
    for node in nodes:
        metadata, status = node["metadata"], node.get("status", {})
        runtime = status.get("nodeInfo", {}).get("containerRuntimeVersion", "")
        ready = any(item["type"] == "Ready" and item["status"] == "True" for item in status.get("conditions", []))
        gpu = int(status.get("allocatable", {}).get("nvidia.com/gpu", "0"))
        simulated = "kwok" in runtime.lower() or metadata.get("labels", {}).get("type") == "kwok"
        inventory.append({"name": metadata["name"], "runtime": runtime, "ready": ready, "allocatable_gpu": gpu, "simulated": simulated})
        if simulated:
            errors.append(f"{metadata['name']}: KWOK cannot execute GPU workloads")
        if not runtime or not ready:
            errors.append(f"{metadata['name']}: real container runtime and Ready condition required")
    if not inventory:
        errors.append("No Kubernetes nodes found")
    if require_gpu and not any(item["allocatable_gpu"] > 0 and item["ready"] and not item["simulated"] for item in inventory):
        errors.append("No Ready real node advertises nvidia.com/gpu")
    return {"passed": not errors, "nodes": inventory, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-device-plugin", action="store_true")
    parser.add_argument("--min-free-gib", type=float, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"passed": False, "errors": [], "nodes": []}
    try:
        response = subprocess.check_output(["kubectl", "get", "nodes", "-o", "json"], text=True, timeout=20)
        result = inspect_nodes(json.loads(response)["items"], not args.before_device_plugin)
        if result["passed"] and args.min_free_gib:
            for node in result["nodes"]:
                raw = subprocess.check_output([
                    "kubectl", "get", "--raw", f"/api/v1/nodes/{node['name']}/proxy/stats/summary"
                ], text=True, timeout=20)
                free = json.loads(raw)["node"]["fs"]["availableBytes"] / 1024**3
                node["free_disk_gib"] = round(free, 2)
                if free < args.min_free_gib:
                    result["errors"].append(f"{node['name']}: {free:.1f} GiB free; need {args.min_free_gib} GiB for cold image/model deployment")
            result["passed"] = not result["errors"]
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        result["errors"].append(str(exc))
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
