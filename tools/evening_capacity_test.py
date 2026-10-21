#!/usr/bin/env python3
"""Duration-based, bounded HTTP/SSE evening workload with explicit SLOs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

try:
    from .realistic_load_test import load_questions, request_one, summarize
except ImportError:
    from realistic_load_test import load_questions, request_one, summarize


def run_phase(url: str, secret: str, questions: list[str], name: str, duration: float,
              concurrency: int, timeout: float, run_id: str, seed: int) -> dict:
    started = time.perf_counter()
    deadline = started + duration

    def client(worker: int) -> list[dict]:
        rng = random.Random(seed + worker)
        results, previous, sequence = [], None, 0
        while time.perf_counter() < deadline:
            replay = previous is not None and sequence % 10 == 9
            if replay:
                plan = previous
            else:
                index = worker + sequence * concurrency
                plan = (f"{run_id}-{name}", index, rng.choice(questions), sequence % 4 == 0,
                        f"{run_id}-{name}-{worker}-{sequence}")
            result = request_one(url, secret, *plan, timeout, require_verified=True, identity_namespace=run_id)
            result["replay"] = replay
            results.append(result)
            if result["business_success"] and not replay:
                previous = plan
            sequence += 1
        return results

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = [item for group in pool.map(client, range(concurrency)) for item in group]
    elapsed = time.perf_counter() - started
    summary = summarize(results)
    ttft = sorted(item["ttft_ms"] for item in results if item.get("ttft_ms") is not None)
    summary["ttft_samples"] = len(ttft)
    summary["ttft_p95_ms"] = ttft[min(len(ttft) - 1, int((len(ttft) - 1) * .95))] if ttft else None
    return {
        "name": name, "duration_seconds": duration, "elapsed_including_drain_seconds": elapsed,
        "concurrency": concurrency, "summary": summary,
        "completed_answers_per_second": sum(item["business_success"] for item in results) / elapsed,
        "replay_requests": sum(item["replay"] for item in results),
        "replay_successes": sum(item["replay"] and item["business_success"] for item in results),
        "sse_requests": sum(item["stream"] for item in results),
    }


def acceptance(stage: dict, min_rps: float, max_p95_ms: float, max_rejection: float) -> dict:
    summary = stage["summary"]
    total = max(summary["requests"], 1)
    rejection = summary["rate_limited"] / total
    failures = 1 - summary["business_success_rate"] - rejection
    answer_p95 = summary.get("successful_answer_latency_ms", {}).get("p95")
    checks = {
        "nonempty_workload": summary["requests"] > 0,
        "answer_throughput": stage["completed_answers_per_second"] >= min_rps,
        "p95_latency": answer_p95 is not None and answer_p95 <= max_p95_ms,
        "business_success_rate": summary["business_success_rate"] >= .99,
        "rejection_budget": rejection <= max_rejection,
        "business_failure_budget": failures <= .01 + 1e-9,
        "sse_completed": summary["sse_complete"] == summary["sse_http_accepted"],
        "incremental_stream_observed": summary["sse_with_incremental_tokens"] > 0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/math_benchmark_1000.csv"))
    parser.add_argument("--steady-seconds", type=float, default=900)
    parser.add_argument("--burst-seconds", type=float, default=180)
    parser.add_argument("--steady-concurrency", type=int, default=50)
    parser.add_argument("--burst-concurrency", type=int, default=300)
    parser.add_argument("--min-steady-rps", type=float, required=True)
    parser.add_argument("--max-p95-ms", type=float, default=20000)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing evidence; select a new output filename")
    if min(args.steady_seconds, args.burst_seconds, args.timeout, args.min_steady_rps, args.max_p95_ms) <= 0 or min(args.steady_concurrency, args.burst_concurrency) < 1:
        parser.error("Duration, concurrency, throughput and latency limits must be positive")
    secret = os.environ.get("AUTH_SHARED_SECRET", "")
    if not secret:
        parser.error("Set AUTH_SHARED_SECRET in the environment; do not put credentials in process arguments")
    questions = load_questions(args.dataset, args.seed, 1000)
    if not questions:
        parser.error("The benchmark dataset is empty")
    run_id = uuid.uuid4().hex
    stages = []
    for name, duration, concurrency in [("steady", args.steady_seconds, args.steady_concurrency), ("burst", args.burst_seconds, args.burst_concurrency)]:
        stage = run_phase(args.url, secret, questions, name, duration, concurrency, args.timeout, run_id, args.seed)
        stage["acceptance"] = acceptance(stage, args.min_steady_rps, args.max_p95_ms, .01)
        stages.append(stage)
        print(json.dumps(stage), flush=True)
    report = {
        "report_type": "duration_based_evening_capacity", "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id, "target": args.url, "stages": stages,
        "configuration": {
            "source_sha256": {
                name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                for name in ("app.py", "config.py", "agentic_rag/skill_handlers.py",
                             "tools/realistic_load_test.py", "tools/evening_capacity_test.py")
            },
            "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            "workload_sha256": hashlib.sha256(json.dumps(questions, ensure_ascii=False).encode()).hexdigest(),
            "force_llm_every_turn": os.getenv("FORCE_LLM_EVERY_TURN", "true"),
            "model": os.getenv("LLM_MODEL_NAME", "unknown"),
            "answer_cache_version": os.getenv("ANSWER_CACHE_VERSION", "default"),
            "ai_max_inflight": os.getenv("AI_MAX_INFLIGHT", "default"),
            "ai_queue_maxsize": os.getenv("AI_QUEUE_MAXSIZE", "default"),
            "isolated_identity_namespace": True,
        },
        "phase_slos_passed": all(stage["acceptance"]["passed"] for stage in stages),
        "full_profile_executed": args.steady_seconds >= 900 and args.burst_seconds >= 180 and args.steady_concurrency >= 50 and args.burst_concurrency >= 300,
        "passed": all(stage["acceptance"]["passed"] for stage in stages) and args.steady_seconds >= 900 and args.burst_seconds >= 180 and args.steady_concurrency >= 50 and args.burst_concurrency >= 300,
        "slo": {"min_completed_answers_per_second": args.min_steady_rps, "max_p95_ms": args.max_p95_ms, "min_business_success_rate": .99, "max_rejection_rate": .01, "verified_answer_required": True},
        "workload": "closed-loop bounded clients; 25% new requests SSE; every tenth request replays the last successful request",
        "limitations": ["Client TTFT is end-to-end first answer delta, not model-only TTFT.",
                        "Failures, HTTP 429 and SSE errors are reported separately from completed answers.",
                        "A passed workload does not establish multi-node HA or fault recovery."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
