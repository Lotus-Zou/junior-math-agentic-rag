#!/usr/bin/env python3
"""Collect real GPU Pod, inference, resource and custom-metric evidence."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    from .gpu_cluster_preflight import inspect_nodes
except ImportError:
    from gpu_cluster_preflight import inspect_nodes


POD_CHECK = """
import json,os,subprocess,urllib.request
base='http://127.0.0.1:8000'
headers={'Authorization':'Bearer '+os.environ['VLLM_API_KEY'],'Content-Type':'application/json'}
request=urllib.request.Request(base+'/v1/chat/completions',headers=headers,data=json.dumps({
 'model':'math-agent','messages':[{'role':'user','content':'Solve 2*x+3=11. Give the answer briefly.'}],
 'max_tokens':64,'temperature':0
}).encode())
with urllib.request.urlopen(request,timeout=90) as response: result=json.load(response)
assert result['choices'][0]['message'].get('content'), 'No actual generated answer'
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
with urllib.request.urlopen(base+'/metrics',timeout=10) as response: metrics=response.read().decode()
assert 'vllm:num_requests_waiting' in metrics, 'Missing queue metric'
print(json.dumps({'gpu':gpu.strip(),'model':result['model'],'usage':result.get('usage'),
 'answer':result['choices'][0]['message']['content'],'queue_metric_present':True}))
"""


def kubectl(*args: str) -> dict:
    return json.loads(subprocess.check_output(["kubectl", *args], text=True, timeout=120))


def verify() -> dict:
    result = inspect_nodes(kubectl("get", "nodes", "-o", "json")["items"])
    if not result["passed"]:
        return result
    result["model_pods"] = []
    pods = kubectl("get", "pods", "-n", "math-agent", "-l", "app=math-agent-model", "-o", "json")["items"]
    if not pods:
        result["errors"].append("No model pods")
    for pod in pods:
        name = pod["metadata"]["name"]
        if pod["metadata"].get("deletionTimestamp"):
            continue
        if not any(c["type"] == "Ready" and c["status"] == "True" for c in pod["status"].get("conditions", [])):
            result["errors"].append(f"{name}: model pod is not Ready")
            continue
        evidence = kubectl("exec", "-n", "math-agent", name, "-c", "vllm", "--", "python3", "-c", POD_CHECK)
        evidence.update({"pod": name, "node": pod["spec"]["nodeName"], "containers": pod["status"].get("containerStatuses", [])})
        result["model_pods"].append(evidence)
    if not result["model_pods"]:
        result["errors"].append("No ready GPU pod executed inference")
    metrics = kubectl("get", "--raw", "/apis/metrics.k8s.io/v1beta1/nodes")
    custom = kubectl("get", "--raw", "/apis/custom.metrics.k8s.io/v1beta1/namespaces/math-agent/pods/*/vllm_num_requests_waiting")
    result["node_metrics"] = metrics.get("items", [])
    result["queue_metrics"] = custom.get("items", [])
    if not result["node_metrics"] or not result["queue_metrics"]:
        result["errors"].append("Resource/custom metrics are missing samples")
    result["passed"] = not result["errors"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify()
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
        result = {"passed": False, "errors": [str(exc)]}
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
