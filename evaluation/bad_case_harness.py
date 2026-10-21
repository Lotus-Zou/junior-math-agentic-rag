# -*- coding: utf-8 -*-
"""Offline product regression gate for the bad-case registry and curriculum SLA."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_rag.fast_path import build_fast_response
from agentic_rag.reliability import resolve_failure

REGISTRY = ROOT / "evaluation" / "bad_case_registry.json"
DATASET = ROOT / "evaluation" / "math_benchmark_1000.csv"
REPORT = ROOT / "evaluation" / "reports" / "product_bad_case_summary.json"
CHAOS_CASES = ROOT / "evaluation" / "chaos_cases.yaml"
FORBIDDEN_USER_COPY = (
    "复杂推理服务",
    "未在 8 秒内完成",
    "系统已自动记录为 bad case",
    "知识库没有召回",
    "critic 服务异常",
    "runtimeerror",
    "provider secret",
)


def _run_chaos_gate() -> tuple[int, list[dict]]:
    cases = yaml.safe_load(CHAOS_CASES.read_text(encoding="utf-8"))["cases"]
    failures = []
    for case in cases:
        response = resolve_failure(
            case["query"],
            case.get("language", "zh"),
            [],
            "",
            case["failure_kind"],
            [case["injection"]],
        )
        answer = response.get("answer", "")
        reasons = []
        if response.get("response_type") != case["expected_response_type"]:
            reasons.append("unexpected response type")
        if case["answer_contains"] not in answer:
            reasons.append("answer is not actionable")
        if not response.get("trace_id"):
            reasons.append("trace id missing")
        lowered = answer.lower()
        leaked = [token for token in FORBIDDEN_USER_COPY if token.lower() in lowered]
        if leaked:
            reasons.append(f"forbidden public copy: {', '.join(leaked)}")
        if case["injection"].lower() in lowered:
            reasons.append("injected private detail leaked")
        if reasons:
            failures.append({"case_id": case["id"], "reasons": reasons})
    return len(cases), failures


def run_gate() -> dict:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    with DATASET.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    chaos_count, chaos_failures = _run_chaos_gate()
    latencies, failures = [], []
    for row in rows:
        started = time.perf_counter()
        response = build_fast_response(row["question"], [])
        latencies.append((time.perf_counter() - started) * 1000)
        if not response:
            failures.append({"question_id": row["question_id"], "reason": "no deterministic route"})
        elif not response.get("validation_passed"):
            failures.append({"question_id": row["question_id"], "reason": "local validation failed"})
        elif response.get("metrics", {}).get("tool_calls") != 0:
            failures.append({"question_id": row["question_id"], "reason": "external tool unexpectedly called"})

    sorted_latency = sorted(latencies)
    p95 = sorted_latency[max(0, int(len(sorted_latency) * 0.95) - 1)]
    sla = float(registry["product_sla"]["deterministic_p95_ms"])
    statuses = {}
    for case in registry["cases"]:
        statuses[case["status"]] = statuses.get(case["status"], 0) + 1
    summary = {
        "passed": not failures and not chaos_failures and p95 <= sla,
        "dataset_size": len(rows),
        "handled": len(rows) - len(failures),
        "failures": failures[:50],
        "chaos": {
            "case_count": chaos_count,
            "passed": chaos_count - len(chaos_failures),
            "failures": chaos_failures,
        },
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 3),
            "p95": round(p95, 3),
            "max": round(max(latencies), 3),
            "threshold_p95": sla,
        },
        "registry_statuses": statuses,
        "open_case_ids": [case["id"] for case in registry["cases"] if case["status"] in {"open", "operational_action_required"}],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="产品 bad-case 回归质量门禁")
    parser.add_argument("--report", action="store_true", help="打印完整 JSON 报告")
    args = parser.parse_args()
    summary = run_gate()
    print(json.dumps(summary if args.report else {key: summary[key] for key in ("passed", "dataset_size", "handled", "latency_ms", "registry_statuses", "open_case_ids")}, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
