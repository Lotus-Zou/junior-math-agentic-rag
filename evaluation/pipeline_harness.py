"""Offline end-to-end harness for YAML pipelines."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_rag.skill_runtime.contracts import SkillContext
from agentic_rag.skill_runtime.executor import SkillExecutor
from agentic_rag.skill_runtime.pipeline import PipelineExecutor, PipelineLoader
from agentic_rag.skill_runtime.registry import get_default_registry

CASE_KEYS = {
    "id", "pipeline", "input", "expected_node", "expected", "contains",
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


def _serialized_public_surface(state):
    return json.dumps(state.get("response_render", {}), ensure_ascii=False, sort_keys=True, default=str)


def _forbidden_values(case):
    values = case.get("not_contains", [])
    return [values] if isinstance(values, str) else values


def check_case(case, state):
    reasons = []
    unsupported = sorted(set(case) - CASE_KEYS)
    if unsupported:
        reasons.append(f"unsupported assertion keys: {', '.join(unsupported)}")
    node = state.get(case["expected_node"])
    if node is None:
        reasons.append("expected node has no value")
    else:
        for key, expected in case.get("expected", {}).items():
            if value_at_path(node, key) != expected:
                reasons.append(f"{key}={value_at_path(node, key)!r}")
        for key, expected in case.get("contains", {}).items():
            if expected not in str(value_at_path(node, key)):
                reasons.append(f"{key} missing {expected!r}")
        if "answer_contains" in case and case["answer_contains"] not in str(value_at_path(node, "answer") or ""):
            reasons.append(f"answer missing {case['answer_contains']!r}")
    for path, expected in case.get("expected_paths", {}).items():
        if value_at_path(state, path) != expected:
            reasons.append(f"{path}={value_at_path(state, path)!r}")
    for path in case.get("absent_paths", []):
        exists, _ = path_value(state, path)
        if exists:
            reasons.append(f"{path} should be absent")
    for forbidden in _forbidden_values(case):
        if forbidden in _serialized_public_surface(state):
            reasons.append(f"public response contains {forbidden!r}")
    return reasons


def run() -> dict:
    registry = get_default_registry()
    loader, runner = PipelineLoader(registry), PipelineExecutor(SkillExecutor(registry))
    failures, total = [], 0
    for path in sorted((ROOT / "evaluation" / "pipeline_cases").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            case, total = json.loads(line), total + 1
            pipeline = loader.load(ROOT / "agentic_rag" / "pipelines" / case["pipeline"])
            context = SkillContext(
                request_id=case["id"],
                trace_id=case["id"],
                deadline_at=datetime.now(timezone.utc) + timedelta(seconds=8),
                policy_set={"allow:math.exercise_generate"},
            )
            state = runner.run(pipeline, case["input"], context)
            reasons = check_case(case, state)
            if reasons:
                failures.append({"id": case["id"], "reasons": reasons, "safe_error": state.get("safe_error", "")})
    return {"passed": not failures, "total": total, "failures": failures}


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
