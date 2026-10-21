"""Background consumer for archival, assessment and learning-report tasks."""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
from threading import Event, Thread
from config import TASK_MAX_ATTEMPTS

from agentic_rag import memory
from agentic_rag import domain_store
from agentic_rag.task_queue import claim, complete, fail, next_item, queue_ready, recover_stale_claims, renew_claim
from agentic_rag.distributed_tracing import span


def handle(item: dict) -> dict:
    with span("worker." + str(item.get("kind", "unknown")), carrier=item.get("trace_context", {})):
        return _handle(item)


def _handle(item: dict) -> dict:
    kind = item.get("kind", "unknown")
    payload = item.get("payload") or {}
    user_id = item.get("user_id", "anonymous")
    session_id = item.get("session_id", "anonymous")
    if kind == "mistake_archive":
        text = payload.get("text") or payload.get("question") or ""
        memory_id = memory.add_memory(text, "mistake_pattern", payload.get("importance", 7), user_id=user_id, session_id=session_id, idempotency_key=item.get("task_id", ""))
        return {"archived": bool(memory_id), "memory_id": memory_id}
    if kind == "knowledge_assessment":
        rows = memory.view_memories(200, user_id=user_id, session_id=session_id)
        counts = Counter(row.get("type", "unknown") for row in rows)
        return {"memory_count": len(rows), "by_type": dict(counts), "knowledge_points": payload.get("knowledge_points", [])}
    if kind == "learning_report":
        rows = memory.view_memories(200, user_id=user_id, session_id=session_id)
        by_type = Counter(row.get("type", "unknown") for row in rows)
        report = domain_store.learning_report(user_id, session_id)
        return {"memory_count": len(rows), "mistake_count": report["mistake_count"], "knowledge_gap_count": by_type.get("knowledge_gap", 0), **report}
    if kind == "evaluation":
        from agentic_rag.guardrails import guided_answer_violations
        from agentic_rag.math_validation import deterministic_math_checks

        answer = str(payload.get("answer", ""))
        query = str(payload.get("query", ""))
        checks = deterministic_math_checks(query, answer, 0)
        return {"answer_nonempty": bool(answer.strip()), "math_checks": checks,
                "direct_answer_violations": guided_answer_violations(answer),
                "scope": "local structural and mathematical checks; not RAGAS semantic judging"}
    raise ValueError(f"unsupported task kind: {kind}")


def main() -> None:
    memory.initialize_memory_db()
    domain_store.initialize_domain_schema()
    if not queue_ready():
        raise RuntimeError("task queue Redis is unavailable")
    recover_stale_claims()
    while True:
        Path("/tmp/math-agent-worker.heartbeat").touch()
        recover_stale_claims()
        item = next_item()
        if not item:
            continue
        task_id = item.get("task_id", "")
        stop_heartbeat = Event()
        heartbeat = None
        try:
            if not claim(item):
                continue
            if item["attempts"] > TASK_MAX_ATTEMPTS:
                fail(task_id, "attempt limit reached after worker recovery", item)
                continue

            def maintain_lease():
                while not stop_heartbeat.wait(15):
                    try:
                        if not renew_claim(item):
                            return
                    except Exception:
                        return

            heartbeat = Thread(target=maintain_lease, daemon=True)
            heartbeat.start()
            result = handle(item)
            if not complete(task_id, result, item):
                continue
            print(json.dumps({"event": "learning_task_completed", "task_id": task_id, "kind": item.get("kind"), "result": result}, ensure_ascii=False), flush=True)
        except Exception as exc:
            retried = fail(task_id, type(exc).__name__, item)
            print(json.dumps({"event": "learning_task_failed", "task_id": task_id, "retry": retried, "error": type(exc).__name__}, ensure_ascii=False), flush=True)
        finally:
            stop_heartbeat.set()
            if heartbeat:
                heartbeat.join(timeout=1)


if __name__ == "__main__":
    main()
