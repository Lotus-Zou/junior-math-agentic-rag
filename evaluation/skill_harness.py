"""Offline contract harness for repository-owned Skills."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_rag.skill_runtime.contracts import SkillContext
from agentic_rag.skill_runtime.executor import SkillExecutor
from agentic_rag.skill_runtime.registry import get_default_registry

CASE_KEYS = {
    "id", "skill", "input", "expected_status", "expected", "contains",
    "answer_contains", "expected_paths", "absent_paths", "not_contains",
}


def path_value(value, dotted_path):
    current = value
    for part in dotted_path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return False, None
            current = current[part]
        elif not hasattr(current, part):
            return False, None
        else:
            current = getattr(current, part)
    return True, current


def value_at_path(value, dotted_path):
    return path_value(value, dotted_path)[1]


def _public_surface(actual):
    exists, response = path_value(actual, "response")
    return response if exists else {}


def _serialized_public_surface(actual):
    return json.dumps(_public_surface(actual), ensure_ascii=False, sort_keys=True, default=str)


def _forbidden_values(case):
    values = case.get("not_contains", [])
    return [values] if isinstance(values, str) else values


def check_case(case, status, actual):
    reasons = []
    unsupported = sorted(set(case) - CASE_KEYS)
    if unsupported:
        reasons.append(f"unsupported assertion keys: {', '.join(unsupported)}")
    if status != case["expected_status"]:
        reasons.append(f"status={status}")
    for key, expected in case.get("expected", {}).items():
        if actual.get(key) != expected:
            reasons.append(f"{key}={actual.get(key)!r}")
    for path, expected in case.get("expected_paths", {}).items():
        if value_at_path(actual, path) != expected:
            reasons.append(f"{path}={value_at_path(actual, path)!r}")
    for path in case.get("absent_paths", []):
        exists, _ = path_value(actual, path)
        if exists:
            reasons.append(f"{path} should be absent")
    public_response = _public_surface(actual)
    for forbidden in _forbidden_values(case):
        if forbidden in _serialized_public_surface(actual):
            reasons.append(f"public response contains {forbidden!r}")
    for key, expected in case.get("contains", {}).items():
        if expected not in str(actual.get(key, "")):
            reasons.append(f"{key} missing {expected!r}")
    if "answer_contains" in case:
        answer = value_at_path(public_response, "answer")
        if case["answer_contains"] not in str(answer or ""):
            reasons.append(f"answer missing {case['answer_contains']!r}")
    return reasons


def run() -> dict:
    executor = SkillExecutor(get_default_registry())
    failures, total = [], 0
    for path in sorted((ROOT / "evaluation" / "skill_cases").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            case, total = json.loads(line), total + 1
            context = SkillContext(
                request_id=case["id"], trace_id=case["id"],
                deadline_at=datetime.now(timezone.utc) + timedelta(seconds=8),
            )
            result = executor.execute(case["skill"], case["input"], context, pipeline="math.evaluation@1.0.0")
            actual = result.value.model_dump(mode="json") if result.value else {}
            reasons = check_case(case, result.status.value, actual)
            if reasons:
                failures.append({"id": case["id"], "reasons": reasons})
    return {"passed": not failures, "total": total, "failures": failures}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skill contract and behavior harness")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    report = run()
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.report else None))
    raise SystemExit(0 if report["passed"] else 1)
