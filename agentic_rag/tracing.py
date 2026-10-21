# -*- coding: utf-8 -*-
"""Auditable JSONL tracing and runtime-budget enforcement."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

from config import MAX_AGENT_STEPS, RUN_TIMEOUT_SECONDS, TRACE_PATH

_trace_lock = Lock()


class RuntimeBudgetExceeded(RuntimeError):
    pass


def new_trace(query: str) -> dict:
    now = time.time()
    return {
        "trace_id": str(uuid.uuid4()),
        "trace_events": [{"node": "start", "at": now, "query": query}],
        "metrics": {"tool_calls": 0, "critic_failures": 0, "hallucinations_detected": 0},
        "started_at": now,
        "deadline_at": now + RUN_TIMEOUT_SECONDS,
        "step_count": 0,
        "tool_calls": [],
    }


def check_budget(state: dict, node: str) -> dict:
    step_count = int(state.get("step_count", 0)) + 1
    if step_count > MAX_AGENT_STEPS:
        raise RuntimeBudgetExceeded(f"Agent 超过最大步骤数 {MAX_AGENT_STEPS}，熔断于 {node}")
    if time.time() > float(state.get("deadline_at", time.time() + RUN_TIMEOUT_SECONDS)):
        raise RuntimeBudgetExceeded(f"Agent 超过 {RUN_TIMEOUT_SECONDS}s 时间预算，熔断于 {node}")
    return {"step_count": step_count}


def _safe(value: Any):
    if hasattr(value, "page_content"):
        return {"content": value.page_content[:1000], "metadata": _safe(value.metadata)}
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items() if key not in {"conversation_history"}}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value[:30]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value if not isinstance(value, str) else value[:4000]
    return str(value)[:1000]


def event(state: dict, node: str, kind: str, payload: Any, started_at: float | None = None) -> dict:
    item = {"node": node, "kind": kind, "at": time.time(), "payload": _safe(payload)}
    if started_at is not None:
        item["latency_ms"] = round((time.time() - started_at) * 1000, 2)
    return {"trace_events": [*state.get("trace_events", []), item]}


def persist_trace(state: dict) -> Path:
    trace_dir = Path(TRACE_PATH)
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_id = state.get("trace_id", "unknown")
    path = trace_dir / f"{trace_id}.jsonl"
    with _trace_lock:
        with path.open("w", encoding="utf-8") as file:
            for item in state.get("trace_events", []):
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
        if not state.get("validation_passed", False):
            append_bad_case(
                trace_id=trace_id,
                query=state.get("query", ""),
                category=state.get("bad_case_category", "validation_failure"),
                issues=state.get("validation_issues", []),
                lock=False,
            )
    return path


def append_bad_case(trace_id: str, query: str, category: str, issues: list[str], *, lock: bool = True) -> Path:
    trace_dir = Path(TRACE_PATH)
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / "bad_cases.jsonl"
    payload = {
        "trace_id": trace_id,
        "at": time.time(),
        "category": category,
        "query": _safe(query),
        "issues": _safe(issues),
    }
    if lock:
        with _trace_lock:
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    else:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path
