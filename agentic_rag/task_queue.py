"""Small durable task queue facade.

Redis is used when configured; the in-process queue keeps local development
usable and makes the degradation mode explicit. Heavy work (archiving a
mistake, generating a report, knowledge-point assessment) must not block /ask.
"""
from __future__ import annotations
import json, uuid
from queue import Queue
from typing import Any
from config import REDIS_URL

_local = Queue(maxsize=10_000)
_redis = None
if REDIS_URL:
    try:
        import redis
        _redis = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=1)
        _redis.ping()
    except Exception:
        _redis = None

def enqueue(kind: str, payload: dict[str, Any], *, user_id: str = "", session_id: str = "") -> str:
    task_id = str(uuid.uuid4())
    item = {"task_id": task_id, "kind": kind, "user_id": user_id, "session_id": session_id, "payload": payload}
    if _redis:
        _redis.rpush("math-agent:learning-tasks", json.dumps(item, ensure_ascii=False))
    else:
        _local.put_nowait(item)
    return task_id

def queue_status() -> dict[str, Any]:
    return {"backend": "redis" if _redis else "memory", "depth": int(_redis.llen("math-agent:learning-tasks")) if _redis else _local.qsize()}
