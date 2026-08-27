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


def value_at_path(value, dotted_path):
    current = value
    for part in dotted_path.split("."):
        current = current.get(part) if isinstance(current, dict) else getattr(current, part, None)
    return current


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
            reasons = []
            if result.status.value != case["expected_status"]:
                reasons.append(f"status={result.status.value}")
            for key, expected in case.get("expected", {}).items():
                if actual.get(key) != expected:
                    reasons.append(f"{key}={actual.get(key)!r}")
            for path, expected in case.get("expected_paths", {}).items():
                if value_at_path(actual, path) != expected:
                    reasons.append(f"{path}={value_at_path(actual, path)!r}")
            for path in case.get("absent_paths", []):
                if value_at_path(actual, path) is not None:
                    reasons.append(f"{path} should be absent")
            for path, forbidden_values in case.get("not_contains", {}).items():
                for forbidden in forbidden_values:
                    if forbidden in str(value_at_path(actual, path)):
                        reasons.append(f"{path} contains {forbidden!r}")
            for key, expected in case.get("contains", {}).items():
                if expected not in str(actual.get(key, "")):
                    reasons.append(f"{key} missing {expected!r}")
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
