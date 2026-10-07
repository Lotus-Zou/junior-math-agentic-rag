"""Background consumer for non-interactive learning tasks.

The worker is intentionally separate from the HTTP process so report
generation and mistake archival cannot consume the AI request pool.
"""
from __future__ import annotations
import json, time
from agentic_rag.task_queue import _redis, _local

def handle(item: dict) -> None:
    # Domain handlers can be replaced by Celery/Temporal consumers without
    # changing the gateway contract.  Keep the acknowledgement boundary here.
    kind = item.get("kind", "unknown")
    print(json.dumps({"event": "learning_task_completed", "task_id": item.get("task_id"), "kind": kind}, ensure_ascii=False), flush=True)

def main() -> None:
    while True:
        item = None
        if _redis:
            raw = _redis.blpop("math-agent:learning-tasks", timeout=5)
            if raw: item = json.loads(raw[1])
        else:
            try: item = _local.get(timeout=5)
            except Exception: pass
        if item:
            try: handle(item)
            except Exception as exc: print(json.dumps({"event":"learning_task_failed","task_id":item.get("task_id"),"error":type(exc).__name__}), flush=True)

if __name__ == "__main__": main()
