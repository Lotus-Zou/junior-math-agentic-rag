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
            context = SkillContext(request_id=case["id"], trace_id=case["id"], deadline_at=datetime.now(timezone.utc) + timedelta(seconds=8))
            state = runner.run(pipeline, case["input"], context)
            node = state.get(case["expected_node"])
            reason = ""
            if node is None:
                reason = "expected node has no value"
            elif case.get("answer_contains") and case["answer_contains"] not in str(getattr(node, "answer", "")):
                reason = f"answer missing {case['answer_contains']!r}"
            if reason:
                failures.append({"id": case["id"], "reason": reason, "safe_error": state.get("safe_error", "")})
    return {"passed": not failures, "total": total, "failures": failures}


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
