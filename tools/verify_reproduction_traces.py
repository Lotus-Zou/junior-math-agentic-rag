#!/usr/bin/env python3
"""Assert real W3C parent links through gateway, gRPC skills and MQ worker."""

import json
from pathlib import Path
import sys
import time
import uuid

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.reproduce_experiments import BASE, headers, write


def main():
    trace_id = uuid.uuid4().hex
    parent = uuid.uuid4().hex[:16]
    user = session = "trace-experiment-" + trace_id
    identity = headers(user, session)
    identity["traceparent"] = f"00-{trace_id}-{parent}-01"
    with httpx.Client(base_url=BASE, timeout=60, trust_env=False) as client:
        ask = client.post("/ask", headers=identity, json={"query":"换个问题", "user_id":user, "session_id":session, "request_id":uuid.uuid4().hex})
        task = client.post("/tasks", headers=identity, json={"kind":"evaluation", "user_id":user, "session_id":session, "payload":{"query":"2x=8","answer":"x=4"}})
        task_id = task.json()["task_id"]
        for _ in range(100):
            status = client.get("/tasks/"+task_id,headers=identity).json()
            if status.get("status") in {"done","failed"}:
                break
            time.sleep(.1)
    spans = []
    for path in (ROOT / ".runtime/traces").glob("spans-*.jsonl"):
        spans.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    spans = [item for item in spans if item["trace_id"] == trace_id]
    gateways = []
    for line in (ROOT / ".runtime/gateway.log").read_text().splitlines():
        if '"event":"gateway_request"' in line:
            item = json.loads(line[line.index("{"):])
            if item["trace_id"] == trace_id:
                gateways.append(item)
    ids = {item["span_id"] for item in spans+gateways}
    checks = {"http_ok":ask.status_code==200,"task_done":status.get("status")=="done",
              "grpc_span":any(item["name"]=="grpc.Ask" for item in spans),
              "skill_span":any(item["name"].startswith("skill.") for item in spans),
              "worker_span":any(item["name"]=="worker.evaluation" for item in spans),
              "gateway_parent":any(item["parent_span_id"]==parent for item in gateways),
              "python_parent_links":all(item["parent_span_id"] in ids for item in spans)}
    write("traces", {"summary":{"passed":all(checks.values()),"checks":checks}, "trace_id":trace_id, "spans":spans,"gateway_spans":gateways})
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
