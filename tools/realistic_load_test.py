#!/usr/bin/env python3
"""Exercise the C-end gateway with a repeatable evening-study workload mix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import random
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


def load_questions(path: Path, seed: int, count: int) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rng = random.Random(seed)
    rng.shuffle(rows)
    return [row["question"].strip() for row in rows[:count]]


def system_snapshot() -> dict:
    try:
        output = subprocess.check_output(
            ["ps", "-eo", "comm=,rss=,pcpu="], text=True, timeout=2
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    watched = {"uvicorn", "gateway", "python", "mariadbd", "redis-server", "ollama"}
    snapshot: dict[str, dict[str, float]] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 3 or parts[0] not in watched:
            continue
        name, rss, cpu = parts
        item = snapshot.setdefault(name, {"rss_mb": 0.0, "cpu_percent_sum": 0.0, "processes": 0})
        item["rss_mb"] += int(rss) / 1024
        item["cpu_percent_sum"] += float(cpu)
        item["processes"] += 1
    return snapshot


def redis_snapshot(redis_url: str) -> dict:
    try:
        import redis

        client = redis.Redis.from_url(redis_url, decode_responses=True)
        info = client.info("stats")
        memory = client.info("memory")
        return {
            "connected_clients": info.get("connected_clients"),
            "total_commands_processed": info.get("total_commands_processed"),
            "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec"),
            "used_memory_mb": round(memory.get("used_memory", 0) / 1024 / 1024, 2),
        }
    except Exception as exc:  # pragma: no cover - depends on external Redis
        return {"error": type(exc).__name__}


def answer_complete(payload: dict, require_verified: bool = False) -> bool:
    if not isinstance(payload, dict) or not payload.get("answer"):
        return False
    if payload.get("response_type") in {"clarification_required", "supported_refusal", "error", "fallback"}:
        return False
    if payload.get("validation_passed") is not True:
        return False
    return not require_verified or (payload.get("response_type") == "verified_answer" and payload.get("validation_passed") is True)


def execution_metadata(payload: dict) -> dict:
    metrics = payload.get("metrics") or {}
    return {"cached": payload.get("cached") is True,
            "model_attempts": int(metrics.get("model_attempts") or 0),
            "execution_path": metrics.get("execution_path", "unknown")}


def read_sse(response, started: float, require_verified: bool = False) -> dict:
    result = {"events": [], "bytes": 0, "delta_events": 0, "business_success": False}
    event, data = "message", []
    final_answer, done, failed = False, False, False

    def dispatch() -> None:
        nonlocal final_answer, done, failed
        if not data:
            return
        result["events"].append(event)
        result.setdefault("first_event_ms", (time.perf_counter() - started) * 1000)
        try:
            payload = json.loads("\n".join(data))
        except json.JSONDecodeError:
            failed = True
            return
        if not isinstance(payload, dict):
            failed = True
            return
        if event in {"answer", "delta", "token"} and payload.get("delta"):
            result.setdefault("ttft_ms", (time.perf_counter() - started) * 1000)
            result["delta_events"] += 1
        if event == "result":
            result.update(execution_metadata(payload))
            result["response_type"] = payload.get("response_type")
            result["validation_passed"] = payload.get("validation_passed")
            final_answer = answer_complete(payload, require_verified)
        if event == "error" or (event == "done" and payload.get("ok") is False):
            failed = True
        if event == "done":
            done = True

    for raw in response:
        result["bytes"] += len(raw)
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            dispatch()
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    dispatch()
    result["business_success"] = final_answer and done and not failed
    return result


def request_one(
    base_url: str,
    secret: str,
    phase: str,
    index: int,
    query: str,
    stream: bool,
    idempotency_key: str,
    timeout: float,
    require_verified: bool = False,
    identity_namespace: str = "",
) -> dict:
    suffix = f"-{identity_namespace}" if identity_namespace else ""
    user_id = f"evening-user-{index % 100}{suffix}"
    session_id = f"evening-session-{index % 40}{suffix}"
    payload = {
        "request_id": f"{phase}-{index}",
        "user_id": user_id,
        "session_id": session_id,
        "query": query,
        "language": "zh",
        "conversation_history": [],
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), user_id.encode("utf-8"), hashlib.sha256).hexdigest()
    endpoint = base_url.rstrip("/")
    if stream and endpoint.endswith("/ask"):
        endpoint += "/stream"
    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Accept": "text/event-stream" if stream else "application/json",
            "Content-Type": "application/json",
            "X-User-ID": user_id,
            "X-Session-ID": session_id,
            "X-User-Signature": signature,
            "Idempotency-Key": idempotency_key,
        },
    )
    started = time.perf_counter()
    result = {
        "phase": phase,
        "index": index,
        "stream": stream,
        "user_id": user_id,
        "status": 0,
        "latency_ms": 0.0,
        "bytes": 0,
        "events": [],
        "business_success": False,
    }
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result["status"] = response.status
            if stream:
                result.update(read_sse(response, started, require_verified))
            else:
                content = response.read()
                result["bytes"] = len(content)
                payload = json.loads(content)
                if isinstance(payload, dict):
                    result.update(execution_metadata(payload))
                result["response_type"] = payload.get("response_type") if isinstance(payload, dict) else None
                result["business_success"] = answer_complete(payload, require_verified)
    except urllib.error.HTTPError as exc:
        content = exc.read()
        result["status"] = exc.code
        result["bytes"] = len(content)
    except Exception as exc:
        result["error"] = type(exc).__name__
    result["latency_ms"] = (time.perf_counter() - started) * 1000
    return result


def run_batch(plans: list[dict], base_url: str, secret: str, concurrency: int, timeout: float) -> list[dict]:
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [
            pool.submit(
                request_one,
                base_url,
                secret,
                plan["phase"],
                plan["index"],
                plan["query"],
                plan["stream"],
                plan["idempotency_key"],
                timeout,
            )
            for plan in plans
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def summarize(results: list[dict]) -> dict:
    latencies = [float(item["latency_ms"]) for item in results]
    statuses: dict[str, int] = {}
    for item in results:
        statuses[str(item["status"])] = statuses.get(str(item["status"]), 0) + 1
    ordered = sorted(latencies)
    successful_latencies = sorted(
        float(item["latency_ms"]) for item in results if item.get("business_success", False)
    )

    def percentile(ratio: float) -> float:
        if not ordered:
            return 0.0
        return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * ratio)))]

    successful = sum(200 <= int(item["status"]) < 300 for item in results)
    business_successful = sum(item.get("business_success", False) for item in results)
    return {
        "requests": len(results),
        "status_counts": statuses,
        "success_rate": successful / max(1, len(results)),
        "business_success_rate": business_successful / max(1, len(results)),
        "answer_cache_hits": sum(item.get("cached", False) and not item.get("replay", False) for item in results),
        "replay_requests": sum(item.get("replay", False) for item in results),
        "uncached_model_attempts": sum(item.get("model_attempts", 0) for item in results
                                       if not item.get("cached") and not item.get("replay")),
        "rate_limited": statuses.get("429", 0),
        "latency_ms": {
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": max(latencies) if latencies else 0.0,
            "mean": statistics.mean(latencies) if latencies else 0.0,
        },
        "successful_answer_latency_ms": {
            "samples": len(successful_latencies),
            "p95": successful_latencies[min(len(successful_latencies)-1, int(round((len(successful_latencies)-1)*.95)))] if successful_latencies else None,
        },
        "sse_complete": sum(
            bool(item.get("stream") and item.get("business_success")) for item in results
        ),
        "sse_with_incremental_tokens": sum(item.get("delta_events", 0) > 1 for item in results),
        "sse_http_accepted": sum(bool(item.get("stream")) and 200 <= int(item["status"]) < 300 for item in results),
        "errors": [item.get("error") for item in results if item.get("error")],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18081/ask")
    parser.add_argument("--secret", required=True)
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/math_benchmark_1000.csv"))
    parser.add_argument("--steady-requests", type=int, default=24)
    parser.add_argument("--steady-concurrency", type=int, default=8)
    parser.add_argument("--burst-requests", type=int, default=48)
    parser.add_argument("--burst-concurrency", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--recovery-delay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--redis-url", default="redis://127.0.0.1:16379/1")
    parser.add_argument("--output", type=Path, default=Path("evaluation/reports/realistic-evening-load.json"))
    args = parser.parse_args()

    questions = load_questions(args.dataset, args.seed, args.steady_requests + args.burst_requests + 20)
    before = {"system": system_snapshot(), "redis": redis_snapshot(args.redis_url)}
    started = time.perf_counter()
    all_results: list[dict] = []
    stages: list[dict] = []

    def stage(name: str, plans: list[dict], concurrency: int) -> None:
        stage_started = time.perf_counter()
        results = run_batch(plans, args.url, args.secret, concurrency, args.timeout)
        all_results.extend(results)
        stages.append({
            "name": name,
            "elapsed_seconds": time.perf_counter() - stage_started,
            "concurrency": concurrency,
            "summary": summarize(results),
        })

    warmup = [
        {"phase": "warmup", "index": i, "query": questions[i], "stream": False, "idempotency_key": f"warmup-{i}"}
        for i in range(min(4, len(questions)))
    ]
    stage("warmup", warmup, 2)

    steady = [
        {
            "phase": "steady",
            "index": i,
            "query": questions[i],
            "stream": i % 5 == 0,
            "idempotency_key": f"steady-{i}",
        }
        for i in range(args.steady_requests)
    ]
    stage("steady_evening", steady, args.steady_concurrency)

    burst = [
        {
            "phase": "burst",
            "index": args.steady_requests + i,
            "query": questions[args.steady_requests + i],
            "stream": i % 5 == 0,
            "idempotency_key": f"burst-{i}",
        }
        for i in range(args.burst_requests)
    ]
    stage("peak_burst", burst, args.burst_concurrency)

    replay_plan = {
        "phase": "idempotency_replay",
        "index": 9001,
        "query": questions[0],
        "stream": False,
        "idempotency_key": "replay-same-key",
    }
    replay_first = run_batch([replay_plan], args.url, args.secret, 1, args.timeout)[0]
    replay_second = run_batch([replay_plan], args.url, args.secret, 1, args.timeout)[0]
    all_results.extend([replay_first, replay_second])
    conflict_plan = dict(replay_plan)
    conflict_plan["phase"] = "idempotency_conflict"
    # Keep the same user as the replay so the gateway sees the same logical key.
    conflict_plan["index"] = 9101
    conflict_plan["query"] = questions[1] + " 请换一种方法。"
    conflict = run_batch([conflict_plan], args.url, args.secret, 1, args.timeout)[0]
    all_results.append(conflict)
    stages.append({
        "name": "idempotency_checks",
        "concurrency": 1,
        "summary": summarize([replay_first, replay_second, conflict]),
        "checks": {
            "replay_statuses": [replay_first["status"], replay_second["status"]],
            "replay_latency_ms": [replay_first["latency_ms"], replay_second["latency_ms"]],
            "conflict_status": conflict["status"],
        },
    })

    if args.recovery_delay > 0:
        cooldown_started = time.perf_counter()
        time.sleep(args.recovery_delay)
        stages.append({
            "name": "breaker_cooldown",
            "elapsed_seconds": time.perf_counter() - cooldown_started,
            "delay_seconds": args.recovery_delay,
        })

    recovery_start = args.steady_requests + args.burst_requests
    recovery = [
        {
            "phase": "recovery",
            "index": recovery_start + i,
            "query": questions[recovery_start + i],
            "stream": i % 4 == 0,
            "idempotency_key": f"recovery-{i}",
        }
        for i in range(8)
    ]
    stage("recovery", recovery, 4)

    report = {
        "report_type": "realistic_evening_load_test",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "target": args.url,
        "identity_mode": "HMAC headers enabled; gateway instance configured with REQUIRE_USER_IDENTITY=true",
        "dataset": str(args.dataset),
        "seed": args.seed,
        "workload": {
            "steady_requests": args.steady_requests,
            "steady_concurrency": args.steady_concurrency,
            "burst_requests": args.burst_requests,
            "burst_concurrency": args.burst_concurrency,
            "sse_mix": "20% steady/burst, 25% recovery",
            "users": 100,
            "sessions": 40,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "stages": stages,
        "aggregate": summarize(all_results),
        "after": {"system": system_snapshot(), "redis": redis_snapshot(args.redis_url)},
        "before": before,
        "limitations": [
            "This run targets the local production-mode Gateway and real local model, Redis and MariaDB processes.",
            "It is not Kubernetes Pod-level capacity evidence and does not establish production peak SLOs.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
