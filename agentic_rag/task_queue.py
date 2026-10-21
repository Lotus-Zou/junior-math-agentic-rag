"""Durable learning-task queue with status, retry and dead-letter tracking."""
from __future__ import annotations

import json
import os
import time
import uuid
from queue import Empty, Queue
from threading import Lock
from typing import Any

from config import REDIS_URL, TASK_MAX_ATTEMPTS
from agentic_rag.distributed_tracing import trace_headers
from redis.exceptions import RedisError

QUEUE_KEY = "math-agent:learning-tasks:v2"
DEAD_LETTER_KEY = "math-agent:learning-tasks:dead:v2"
CONSUMER_GROUP = os.getenv("TASK_CONSUMER_GROUP", "math-agent-workers")
CONSUMER_NAME = os.getenv("TASK_CONSUMER_NAME", f"worker-{os.getpid()}-{uuid.uuid4().hex[:8]}")
TASK_RETENTION_SECONDS = 7 * 86400
TASK_QUEUE_CAPACITY = max(1, min(100_000, int(os.getenv("TASK_QUEUE_CAPACITY", "10000"))))
_recovery_cursor = "0-0"
_local = Queue(maxsize=10_000)
_recovered = Queue(maxsize=10_000)
_local_tasks: dict[str, dict[str, Any]] = {}
_local_lock = Lock()


class QueueCapacityExceeded(RuntimeError):
    """The queue cannot accept another task without exceeding its budget."""


class QueueUnavailable(RuntimeError):
    """The configured durable queue cannot accept or retrieve tasks."""


_redis = None
_redis_required = bool(REDIS_URL)
if REDIS_URL:
    try:
        import redis
        # XREADGROUP uses a blocking read; its socket timeout must exceed the
        # BLOCK interval or an idle worker exits every polling cycle.
        _redis = redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=10,
        )
    except Exception:
        _redis = None


def _ensure_group() -> None:
    if not _redis:
        return
    try:
        _redis.xgroup_create(QUEUE_KEY, CONSUMER_GROUP, id="0-0", mkstream=True)
    except Exception as exc:
        if "BUSYGROUP" not in str(exc):
            raise


def queue_ready() -> bool:
    if not _redis_required:
        return True
    if _redis is None:
        return False
    try:
        return bool(_redis.ping())
    except Exception:
        return False


def _key(task_id: str) -> str:
    return f"math-agent:task:{task_id}"


def enqueue(kind: str, payload: dict[str, Any], *, user_id: str = "", session_id: str = "") -> str:
    task_id = str(uuid.uuid4())
    item = {"task_id": task_id, "kind": kind, "user_id": user_id or "anonymous", "session_id": session_id or "anonymous", "payload": payload, "status": "queued", "attempts": 0, "created_at": time.time(), "updated_at": time.time(), "result": None, "error": ""}
    item["trace_context"] = trace_headers()
    if _redis_required and not _redis:
        raise QueueUnavailable("task queue Redis is unavailable")
    if _redis:
        # Admission and publication must be atomic across gateway replicas.
        # Never trim an active stream: that would discard unprocessed work.
        script = """
        if redis.call('XLEN',KEYS[1])>=tonumber(ARGV[1]) then return 0 end
        redis.call('HSET',KEYS[2],unpack(ARGV,4))
        redis.call('EXPIRE',KEYS[2],ARGV[2])
        redis.call('XADD',KEYS[1],'*','task',ARGV[3])
        return 1
        """
        fields = []
        for key, value in item.items():
            encoded = json.dumps(value, ensure_ascii=False) if value is None or isinstance(value, (dict, list)) else str(value)
            fields.extend((key, encoded))
        try:
            _ensure_group()
            accepted = _redis.eval(script, 2, QUEUE_KEY, _key(task_id), TASK_QUEUE_CAPACITY,
                                   TASK_RETENTION_SECONDS, json.dumps(item, ensure_ascii=False), *fields)
        except RedisError as exc:
            raise QueueUnavailable("task queue Redis is unavailable") from exc
        if not accepted:
            raise QueueCapacityExceeded("task queue capacity reached")
    else:
        with _local_lock:
            cutoff = time.time() - TASK_RETENTION_SECONDS
            expired = [key for key, task in _local_tasks.items()
                       if task["status"] in {"done", "failed"} and task["updated_at"] < cutoff]
            for key in expired:
                del _local_tasks[key]
            active = sum(task["status"] in {"queued", "running"} for task in _local_tasks.values())
            if active >= min(TASK_QUEUE_CAPACITY, _local.maxsize):
                raise QueueCapacityExceeded("task queue capacity reached")
            _local.put_nowait(item)
            _local_tasks[task_id] = item
    return task_id


def _decode_hash(value: dict[str, str]) -> dict[str, Any]:
    result = dict(value)
    for field in ("payload", "result", "trace_context"):
        if result.get(field) == "null":
            result[field] = None
        elif result.get(field) not in (None, ""):
            result[field] = json.loads(result[field])
    result["attempts"] = int(result.get("attempts", 0))
    for field in ("created_at", "updated_at"):
        result[field] = float(result.get(field, 0))
    return result


def get_task(task_id: str) -> dict[str, Any] | None:
    if _redis:
        try:
            raw = _redis.hgetall(_key(task_id))
        except RedisError as exc:
            raise QueueUnavailable("task queue Redis is unavailable") from exc
        return _decode_hash(raw) if raw else None
    if _redis_required:
        raise QueueUnavailable("task queue Redis is unavailable")
    with _local_lock:
        item = _local_tasks.get(task_id)
        return dict(item) if item else None


def claim(item: dict[str, Any]) -> bool:
    task_id = item["task_id"]
    if _redis:
        token = uuid.uuid4().hex
        script = """
        local state=redis.call('HGET',KEYS[1],'status')
        if state=='done' or state=='failed' then return -1 end
        if not redis.call('SET',KEYS[2],ARGV[1],'NX','EX',120) then return 0 end
        local attempts=redis.call('HINCRBY',KEYS[1],'attempts',1)
        redis.call('HSET',KEYS[1],'status','running','updated_at',ARGV[2])
        return attempts
        """
        attempts = int(_redis.eval(script, 2, _key(task_id), _key(task_id) + ":lease", token, time.time()))
        if attempts == -1:
            if item.get("_stream_id"):
                _redis.xack(QUEUE_KEY, CONSUMER_GROUP, item["_stream_id"])
                _redis.xdel(QUEUE_KEY, item["_stream_id"])
            return False
        if attempts == 0:
            return False
        item.update(status="running", attempts=attempts, updated_at=time.time(), _lease_token=token)
        return True
    item["status"] = "running"
    item["attempts"] = int(item.get("attempts", 0)) + 1
    item["updated_at"] = time.time()
    with _local_lock:
        current = _local_tasks.get(task_id, {})
        if current.get("status") in {"done", "failed"}:
            return False
        _local_tasks[task_id] = dict(item)
    return True


def renew_claim(item: dict[str, Any]) -> bool:
    if not _redis or not item.get("_lease_token"):
        return True
    script = "if redis.call('GET',KEYS[1])==ARGV[1] then redis.call('EXPIRE',KEYS[1],120); return 1 end; return 0"
    return bool(_redis.eval(script, 1, _key(item["task_id"]) + ":lease", item["_lease_token"]))


def complete(task_id: str, result: dict[str, Any], item: dict[str, Any] | None = None) -> bool:
    now = time.time()
    if _redis:
        script = """
        if redis.call('GET',KEYS[2])~=ARGV[1] then return 0 end
        redis.call('HSET',KEYS[1],'status','done','result',ARGV[2],'updated_at',ARGV[3])
        redis.call('EXPIRE',KEYS[1],ARGV[4])
        redis.call('DEL',KEYS[2])
        if ARGV[6]~='' then
          redis.call('XACK',KEYS[3],ARGV[5],ARGV[6])
          redis.call('XDEL',KEYS[3],ARGV[6])
        end
        return 1
        """
        return bool(_redis.eval(script, 3, _key(task_id), _key(task_id) + ":lease", QUEUE_KEY,
                                (item or {}).get("_lease_token", ""), json.dumps(result, ensure_ascii=False),
                                now, TASK_RETENTION_SECONDS, CONSUMER_GROUP, (item or {}).get("_stream_id", "")))
    else:
        with _local_lock:
            if task_id in _local_tasks:
                _local_tasks[task_id].update(status="done", result=result, updated_at=now)
        return True


def fail(task_id: str, error: str, item: dict[str, Any]) -> bool:
    now = time.time()
    attempts = int(item.get("attempts", 0))
    retry = attempts < TASK_MAX_ATTEMPTS
    status = "queued" if retry else "failed"
    if _redis:
        retry_item = {key: value for key, value in item.items() if not key.startswith("_")}
        script = """
        if redis.call('GET',KEYS[2])~=ARGV[1] then return 0 end
        redis.call('HSET',KEYS[1],'status',ARGV[2],'error',ARGV[3],'updated_at',ARGV[4])
        redis.call('EXPIRE',KEYS[1],ARGV[8])
        redis.call('DEL',KEYS[2])
        if ARGV[2]=='queued' then
          redis.call('XADD',KEYS[3],'*','task',ARGV[5])
        else
          redis.call('XADD',KEYS[4],'MAXLEN','~',100000,'*','task',ARGV[5])
        end
        if ARGV[7]~='' then
          redis.call('XACK',KEYS[3],ARGV[6],ARGV[7])
          redis.call('XDEL',KEYS[3],ARGV[7])
        end
        return 1
        """
        committed = _redis.eval(script, 4, _key(task_id), _key(task_id) + ":lease", QUEUE_KEY, DEAD_LETTER_KEY,
                                item.get("_lease_token", ""), status, error[:500], now,
                                json.dumps(retry_item, ensure_ascii=False), CONSUMER_GROUP,
                                item.get("_stream_id", ""), TASK_RETENTION_SECONDS)
        if not committed:
            return False
    else:
        with _local_lock:
            if task_id in _local_tasks:
                _local_tasks[task_id].update(status=status, error=error[:500], updated_at=now)
        if retry:
            _local.put_nowait(item)
    return retry


def queue_status() -> dict[str, Any]:
    if _redis:
        _ensure_group()
        pending = _redis.xpending(QUEUE_KEY, CONSUMER_GROUP)
        return {"backend": "redis-stream", "depth": int(_redis.xlen(QUEUE_KEY)), "pending": int(pending.get("pending", 0) if isinstance(pending, dict) else pending[0]), "dead_letter_depth": int(_redis.xlen(DEAD_LETTER_KEY))}
    if _redis_required:
        return {"backend": "unavailable", "depth": 0, "pending": 0, "dead_letter_depth": 0}
    with _local_lock:
        dead = sum(item.get("status") == "failed" for item in _local_tasks.values())
    return {"backend": "memory", "depth": _local.qsize(), "dead_letter_depth": dead}


def next_item(timeout: int = 5) -> dict[str, Any] | None:
    try:
        return _recovered.get_nowait()
    except Empty:
        pass
    if _redis:
        _ensure_group()
        try:
            raw = _redis.xreadgroup(
                CONSUMER_GROUP,
                CONSUMER_NAME,
                {QUEUE_KEY: ">"},
                count=1,
                block=timeout * 1000,
            )
        except TimeoutError:
            return None
        if not raw:
            return None
        _stream, messages = raw[0]
        stream_id, fields = messages[0]
        item = json.loads(fields.get("task", "{}"))
        item["_stream_id"] = stream_id
        return item
    if _redis_required:
        time.sleep(min(timeout, 1))
        return None
    try:
        return _local.get(timeout=timeout)
    except Empty:
        return None


def recover_stale_claims(min_idle_ms: int = 60_000) -> int:
    """Reclaim tasks from workers that died after receiving a stream message."""
    if not _redis:
        return 0
    global _recovery_cursor
    _ensure_group()
    try:
        result = _redis.xautoclaim(QUEUE_KEY, CONSUMER_GROUP, CONSUMER_NAME, min_idle_ms, start_id=_recovery_cursor, count=100)
    except Exception:
        return 0
    messages = result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else []
    _recovery_cursor = result[0] if result else "0-0"
    for stream_id, fields in messages:
        item = json.loads(fields.get("task", "{}"))
        item["_stream_id"] = stream_id
        task_id = item.get("task_id")
        try:
            _recovered.put_nowait(item)
        except Exception:
            # Leave the message pending if local recovery capacity is exhausted.
            continue
    return len(messages)
