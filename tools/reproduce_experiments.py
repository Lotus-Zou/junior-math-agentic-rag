#!/usr/bin/env python3
"""Measured full-stack experiments. Run through reproduction_stack.py run."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import hmac
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import uuid

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.realistic_load_test import answer_complete
REPORTS = ROOT / "evaluation/reports"
PREFIX = os.getenv("REPRO_REPORT_PREFIX", "reconstructed-20261020")
BASE = "http://127.0.0.1:28080"


def write(name, data):
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"{PREFIX}-{name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(path), "summary": data.get("summary", {})}, ensure_ascii=False), flush=True)


def headers(user="experiment-user", session="experiment-session"):
    signature = hmac.new(os.environ["AUTH_SHARED_SECRET"].encode(), user.encode(), hashlib.sha256).hexdigest()
    return {"X-User-ID": user, "X-Session-ID": session, "X-User-Signature": signature}


def experiment_identity(index, namespace=""):
    suffix = f"-{namespace}" if namespace else ""
    return f"experiment-user-{index % 100}{suffix}", f"experiment-session-{index}{suffix}"


def post_question(query, *, index=0, history=None, stream=False, key=None, namespace=""):
    user, session = experiment_identity(index, namespace)
    identity = headers(user, session)
    if key:
        identity["Idempotency-Key"] = key
    body = {"query": query, "user_id": user, "session_id": session, "language": "zh",
            "conversation_history": history or [], "request_id": uuid.uuid4().hex}
    started = time.perf_counter()
    result = {"index": index, "query": query, "stream": stream, "status": 0}
    try:
        with httpx.Client(timeout=60, trust_env=False) as client:
            if stream:
                events = []
                event, data = "message", []
                with client.stream("POST", BASE + "/ask/stream", json=body, headers=identity) as response:
                    result["status"] = response.status_code
                    for line in response.iter_lines():
                        if line.startswith("event:"):
                            event = line[6:].strip()
                        elif line.startswith("data:"):
                            data.append(line[5:].strip())
                        elif not line and data:
                            payload = json.loads("\n".join(data))
                            events.append({"event": event, "elapsed_ms": (time.perf_counter()-started)*1000, "data": payload})
                            event, data = "message", []
                result["events"] = events
                finals = [item["data"] for item in events if item["event"] == "result"]
                payload = finals[-1] if finals else {}
                deltas = [item for item in events if item["data"].get("delta")]
                result["delta_events"] = len(deltas)
                result["ttft_ms"] = deltas[0]["elapsed_ms"] if deltas else None
                result["stream_complete"] = bool(finals and events and events[-1]["event"] == "done" and not any(item["event"] == "error" for item in events))
            else:
                response = client.post(BASE + "/ask", json=body, headers=identity)
                result["status"] = response.status_code
                payload = response.json()
            result["response"] = payload
            result["transport_success"] = result["status"] == 200 and bool(payload.get("answer")) and (not stream or result["stream_complete"])
            result["business_success"] = result["transport_success"] and answer_complete(payload)
            result["verified"] = payload.get("response_type") == "verified_answer" and payload.get("validation_passed") is True
    except Exception as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
        result["business_success"] = False
    result["latency_ms"] = (time.perf_counter()-started)*1000
    return result


def summary(rows):
    ordered = sorted(row["latency_ms"] for row in rows)
    models = Counter()
    for row in rows:
        if row.get("response", {}).get("cached"):
            continue
        metrics = row.get("response", {}).get("metrics", {})
        for key in ("model_attempts", "model_successes", "model_failures"):
            models[key] += int(metrics.get(key) or 0)
    return {"requests": len(rows), "status_counts": dict(Counter(row["status"] for row in rows)),
            "business_successes": sum(row["business_success"] for row in rows),
            "verified_answers": sum(row.get("verified", False) for row in rows),
            "latency_p50_ms": ordered[int((len(ordered)-1)*.5)] if ordered else 0,
            "latency_p95_ms": ordered[int((len(ordered)-1)*.95)] if ordered else 0,
            "model_calls": dict(models), "answer_cache_hits": sum(bool(row.get("response", {}).get("cached")) for row in rows),
            "response_types": dict(Counter(row.get("response", {}).get("response_type", "http_error") for row in rows))}


def smoke():
    checks = {}
    namespace = uuid.uuid4().hex
    with httpx.Client(base_url=BASE, timeout=60, trust_env=False) as browser:
        checks["homepage"] = browser.get("/").status_code == 200
        checks["readiness"] = browser.get("/ready").status_code == 200
        identity = browser.get("/auth/session")
        checks["browser_identity"] = identity.status_code == 200 and bool(identity.json().get("user_id"))
        checks["browser_report"] = browser.get("/reports/learning").status_code == 200
        checks["impersonation_blocked"] = browser.get("/reports/learning", headers={"X-User-ID": "another-student"}).status_code == 401
    first = post_question("解方程 2(x-3)=8。", index=70001, key="smoke-"+uuid.uuid4().hex, namespace=namespace)
    second = post_question("为什么这一步要两边同时除以2？", index=70001,
                           history=first.get("response", {}).get("conversation_history", []), namespace=namespace)
    stream = post_question("一元一次方程移项时为什么要改变符号？请结合等式的性质解释。", index=70002, stream=True, namespace=namespace)
    checks["first_question"] = first["business_success"]
    checks["follow_up"] = second["business_success"]
    checks["sse_completed"] = stream.get("transport_success", False)
    checks["sse_answer_completed"] = stream["business_success"]
    checks["incremental_deltas"] = stream.get("delta_events", 0) > 1
    with httpx.Client(base_url=BASE, timeout=60, trust_env=False) as client:
        idem_headers = headers("smoke-idem", "smoke-idem")
        idem_headers["Idempotency-Key"] = uuid.uuid4().hex
        idem_body = {"user_id":"smoke-idem","session_id":"smoke-idem","query":"解方程 13(x-509)=26。", "request_id":uuid.uuid4().hex}
        with ThreadPoolExecutor(max_workers=2) as pool:
            first_attempt = pool.submit(client.post, "/ask", json=idem_body, headers=idem_headers)
            time.sleep(.03)
            duplicate = client.post("/ask", json=idem_body, headers=idem_headers)
            original = first_attempt.result()
        replay = client.post("/ask", json=idem_body, headers=idem_headers)
        conflict = client.post("/ask", json={**idem_body,"query":"different"},headers=idem_headers)
        duplicate_in_progress = (
            duplicate.status_code == 409
            and duplicate.json().get("code") == "IDEMPOTENCY_IN_PROGRESS"
        )
        duplicate_replayed = (
            duplicate.status_code == 200
            and original.status_code == 200
            and duplicate.content == original.content
        )
        checks["concurrent_idempotency"] = original.status_code == 200 and (
            duplicate_in_progress or duplicate_replayed
        )
        checks["idempotency_replay"] = replay.status_code==200 and replay.content==original.content
        checks["idempotency_conflict"] = conflict.status_code==409 and conflict.json().get("code")=="IDEMPOTENCY_CONFLICT"
        h = headers("smoke-task", "smoke-task")
        tasks = []
        for kind in ("learning_report", "knowledge_assessment", "evaluation", "mistake_archive"):
            response = client.post("/tasks", json={"kind": kind, "user_id": "smoke-task", "session_id": "smoke-task",
                    "payload": {"query": "2x=8", "answer": "x=4", "text": "移项时漏改符号"}}, headers=h)
            if response.status_code != 202:
                checks["task_"+kind] = False
                continue
            task_id = response.json()["task_id"]
            deadline = time.monotonic()+30
            while time.monotonic() < deadline:
                task = client.get("/tasks/"+task_id, headers=h).json()
                if task.get("status") in {"done", "failed"}:
                    break
                time.sleep(.2)
            checks["task_"+kind] = task.get("status") == "done"
            checks["task_ownership_"+kind] = client.get("/tasks/"+task_id, headers=headers("stranger", "smoke-task")).status_code == 404
            tasks.append(task)
        checks["feedback"] = client.post("/feedback", headers=headers(*experiment_identity(70001, namespace)),
                json={"trace_id": first.get("response", {}).get("trace_id", ""), "correct": True, "comment": "reproduction experiment"}).status_code == 200
    write("smoke", {"summary": {"checks": checks, "passed": all(checks.values()),
                               "transport_successes": sum(row.get("transport_success", False) for row in (first, second, stream)),
                               "complete_answers": sum(row["business_success"] for row in (first, second, stream)),
                               "answer_requests": 3},
                    "scope": "Live integration checks; application validation is not independent mathematical ground truth.",
                    "requests": [first, second, stream], "tasks": tasks})
    return all(checks.values())


def quality(limit, workers):
    with (ROOT / "evaluation/math_benchmark_1000.csv").open(encoding="utf-8-sig") as handle:
        cases = list(csv.DictReader(handle))
    random.Random(20261020).shuffle(cases)
    cases = cases[:limit] if limit else cases
    checkpoint = REPORTS / f"{PREFIX}-quality-checkpoint.jsonl"
    previous = [json.loads(line) for line in checkpoint.read_text().splitlines()] if checkpoint.exists() else []
    # A checkpoint is reusable only for the same question content.
    completed = {row["case_id"]: row for row in previous if row["status"] == 200}
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool, checkpoint.open("a") as log:
        futures = {}
        for index, case in enumerate(cases):
            if case["question_id"] in completed and completed[case["question_id"]]["query"] == case["question"]:
                results.append(completed[case["question_id"]])
                continue
            future = pool.submit(post_question, case["question"], index=80000+index)
            futures[future] = case
        for future in as_completed(futures):
            row = future.result()
            case = futures[future]
            row.update(case_id=case["question_id"], reference_answer=case["reference_answer"], family=case["knowledge_point"])
            results.append(row)
            log.write(json.dumps(row, ensure_ascii=False)+"\n")
            log.flush()
            if len(results) % 25 == 0:
                print(json.dumps({"completed": len(results), "total": len(cases), "summary": summary(results)}, ensure_ascii=False), flush=True)
    write("quality", {"summary": summary(results), "checkpoint_previous_failed_attempts": sum(row["status"] != 200 for row in previous),
                       "checkpoint_note": "Resumed successful cases; previous failed attempts remain in the append-only checkpoint and interrupted-run report. This is per-case completion, not uninterrupted availability.",
                       "scope": "Actual gateway responses on unique synthetic domain questions. Verified means application validation, not independent ground-truth or RAGAS correctness.",
                       "workers": workers, "results": results})


def resource_snapshot():
    from tools.reproduction_stack import configuration
    processes = {}
    for name, entry in configuration()["processes"].items():
        try:
            fields = dict(line.split(":",1) for line in Path(f"/proc/{entry['pid']}/status").read_text().splitlines() if ":" in line)
            processes[name] = {"pid":entry["pid"],"rss_kib":int(fields["VmRSS"].split()[0]),"threads":int(fields["Threads"])}
        except (OSError,KeyError):
            processes[name] = {"unavailable":True}
    gpu = subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,memory.used,power.draw","--format=csv,noheader,nounits"], capture_output=True,text=True)
    return {"time":time.time(),"owned_processes":processes,"gpu_csv":gpu.stdout.strip(),"gpu_note":"Device-level usage; includes other users if any. Ollama runner children are not included in the parent RSS values."}


def load(report_label):
    from tools.realistic_load_test import request_one, summarize
    stages = []
    for count, concurrency, label in ((24,4,"steady"), (160,160,"burst"), (8,2,"recovery")):
        if label == "recovery":
            time.sleep(6)
        start = time.perf_counter()
        before = resource_snapshot()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(request_one, BASE+"/ask", os.environ["AUTH_SHARED_SECRET"], label, i,
                        "解方程 2(x-3)=8。" if i % 2 else f"解方程 3(x-{i+300})=15。", i%4==0,
                        f"{uuid.uuid4().hex}", 60) for i in range(count)]
            rows = [future.result() for future in as_completed(futures)]
        elapsed = time.perf_counter()-start
        stage = {"name": label, "concurrency": concurrency, "elapsed_seconds": elapsed,
                 "summary": summarize(rows), "requests": rows, "resources_before":before, "resources_after":resource_snapshot()}
        stage["summary"]["completed_rps"] = count/elapsed
        stage["summary"]["business_rps"] = sum(row["business_success"] for row in rows)/elapsed
        stages.append(stage)
        write(report_label, {"summary": {stage["name"]: stage["summary"] for stage in stages}, "stages": stages,
                       "force_llm_every_turn":os.getenv("FORCE_LLM_EVERY_TURN"),
                       "scope": "Real single-host gateway/gRPC/model/MySQL/Redis. 429 is rejection, not a successful answer; workload mixes one hot key and distinct problems."})


def public_benchmark(limit):
    from evaluation.public_benchmarks import extract_prediction

    source = ROOT / ".runtime/resources/math-eval-TAL-SCQ5K/ch_single_choice_constructed_5K/ch_single_choice_test_2K.jsonl"
    cases = [json.loads(line) for line in source.read_text().splitlines()]
    cases = [case for case in cases if case["dataset_name"].startswith("mid_math_")]
    total = len(cases)
    random.Random(20261020).shuffle(cases)
    cases = cases[:limit] if limit else cases
    results = []
    for index, case in enumerate(cases):
        options = [item for group in case["answer_option_list"] for item in group]
        prompt = case["problem"]+"\n"+"\n".join(item["aoVal"]+". "+item["content"] for item in options)
        prompt += "\n请解释解题思路，最后一行写答案字母，格式为：答案：A。"
        result = post_question(prompt, index=100000+index)
        result["case_id"] = case["queId"]
        result["expected"] = case["answer_value"]
        result["prediction"] = extract_prediction("ceval_middle_school_mathematics", result.get("response", {}).get("answer", ""))
        result["exact_match"] = result["status"] == 200 and result["prediction"] == result["expected"]
        results.append(result)
        write("public-tal", {"summary": {"evaluated":len(results), "available_junior_high":total,
            "exact_matches":sum(item["exact_match"] for item in results), "exact_match_accuracy":sum(item["exact_match"] for item in results)/len(results),
            "unparseable_or_no_answer":sum(not item["prediction"] for item in results)},
            "dataset": "TAL-SCQ5K Chinese TEST, dataset_name mid_math_* (junior-high competition)",
            "selection": "Seeded shuffle (20261020), no cherry-picking; missing figures and refusals remain in denominator.",
            "scope": "Public held-out competition sample; reference and analysis never sent to service. Not representative of all school mistakes.",
            "results":results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "quality", "load", "public"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--label", default="load")
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    if args.mode == "smoke":
        return 0 if smoke() else 1
    elif args.mode == "load":
        load(args.label)
    elif args.mode == "public":
        public_benchmark(args.limit)
    else:
        quality(args.limit, args.workers)


if __name__ == "__main__":
    sys.exit(main())
