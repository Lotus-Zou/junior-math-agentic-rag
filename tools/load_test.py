#!/usr/bin/env python3
"""Small dependency-free HTTP load test for the C-end gateway.

This script measures the deployed stack; it does not manufacture capacity claims.
Run it against the same gateway, model profile, database and Redis topology used
for the release candidate, then archive the JSON output with the release.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def one(url: str, secret: str, index: int, timeout: float) -> dict:
    user_id = f"load-user-{index % 100}"
    session_id = f"load-session-{index}"
    payload = {
        "request_id": f"load-{index}",
        "user_id": user_id,
        "session_id": session_id,
        "query": "解方程 2x + 3 = 11",
        "language": "zh",
        "conversation_history": [],
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), user_id.encode("utf-8"), hashlib.sha256).hexdigest()
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-User-ID": user_id,
            "X-Session-ID": session_id,
            "X-User-Signature": signature,
            "Idempotency-Key": f"load-{index}",
        },
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            status = response.status
        return {"status": status, "latency_ms": (time.perf_counter() - started) * 1000}
    except urllib.error.HTTPError as exc:
        exc.read()
        return {"status": exc.code, "latency_ms": (time.perf_counter() - started) * 1000}
    except Exception as exc:
        return {"status": 0, "latency_ms": (time.perf_counter() - started) * 1000, "error": type(exc).__name__}


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080/ask")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--secret", required=True)
    parser.add_argument("--output", default="load-test-result.json")
    args = parser.parse_args()
    started = time.perf_counter()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [pool.submit(one, args.url, args.secret, index, args.timeout) for index in range(args.requests)]
        for future in as_completed(futures):
            results.append(future.result())
    latencies = [float(item["latency_ms"]) for item in results]
    status_counts: dict[str, int] = {}
    for item in results:
        key = str(item["status"])
        status_counts[key] = status_counts.get(key, 0) + 1
    successful = sum(1 for item in results if 200 <= int(item["status"]) < 300)
    report = {
        "url": args.url,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "elapsed_seconds": time.perf_counter() - started,
        "throughput_rps": len(results) / max(0.001, time.perf_counter() - started),
        "success_rate": successful / max(1, len(results)),
        "status_counts": status_counts,
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies) if latencies else 0,
            "mean": statistics.mean(latencies) if latencies else 0,
        },
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
