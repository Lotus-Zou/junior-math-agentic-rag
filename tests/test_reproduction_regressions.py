import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

import app
from agentic_rag.cache import AnswerCache
from evaluation.generate_dataset import build_rows


def test_timeout_keeps_capacity_until_inference_thread_finishes(monkeypatch):
    release = Event()
    entered = Event()
    executor = ThreadPoolExecutor(max_workers=1)

    def blocked(_request):
        entered.set()
        release.wait(3)

    monkeypatch.setattr(app, "AI_MAX_INFLIGHT", 1)
    monkeypatch.setattr(app, "AI_QUEUE_MAXSIZE", 0)
    monkeypatch.setattr(app, "RUN_TIMEOUT_SECONDS", .02)
    monkeypatch.setattr(app, "_graph_admission", None)
    monkeypatch.setattr(app, "_graph_executor", executor)
    monkeypatch.setattr(app, "_run_curriculum_skill", blocked)

    async def verify():
        request = app.AskRequest(query="解方程 x+1=2")
        with pytest.raises(asyncio.TimeoutError):
            await app._run_bounded_graph(request)
        assert entered.is_set()
        with pytest.raises(app.HTTPException) as error:
            await app._run_bounded_graph(request)
        assert error.value.status_code == 429
        release.set()
        for _ in range(100):
            if app._graph_admission._value == 1:
                break
            await asyncio.sleep(.005)
        assert app._graph_admission._value == 1

    try:
        asyncio.run(verify())
    finally:
        release.set()
        executor.shutdown(wait=True)


def test_dataset_has_unique_underlying_problems_and_no_correct_wrong_labels():
    rows = build_rows()
    assert len(rows) == len({row["problem_text"] for row in rows}) == 1000
    assert all(not row["student_wrong_answer"] or row["student_wrong_answer"] != row["reference_answer"] for row in rows)


def test_shared_cache_outage_preserves_local_hits_and_allows_misses():
    class BrokenRedis:
        def get(self, *_args):
            raise ConnectionError("simulated Redis loss")

        def setex(self, *_args):
            raise ConnectionError("simulated Redis loss")

    cache = AnswerCache()
    cache.redis = BrokenRedis()
    cache.set({"query": "cached"}, {"answer": "verified"})
    assert cache.get({"query": "cached"}) == {"answer": "verified"}
    assert cache.get({"query": "uncached"}) is None


def test_response_cache_ignores_transport_metadata_but_retains_owner_and_history():
    request = app.AskRequest(query="x+1=2", user_id="student-1", session_id="session-1", request_id="request-1")
    equivalent = request.model_copy(update={"request_id": "request-2", "stream_output": True})
    canonical = lambda value: value.model_dump(exclude={"request_id", "stream_output"})
    assert AnswerCache.key(canonical(request)) == AnswerCache.key(canonical(equivalent))
    assert AnswerCache.key(canonical(request)) != AnswerCache.key(canonical(request.model_copy(update={"user_id": "student-2"})))


def test_skill_timeout_retains_global_slot_without_overlapping_retry(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from threading import BoundedSemaphore
    from types import SimpleNamespace
    from pydantic import BaseModel
    from agentic_rag.skill_runtime import executor as runtime
    from agentic_rag.skill_runtime.contracts import SkillContext, SkillStatus

    release, entered = Event(), Event()
    pool = ThreadPoolExecutor(max_workers=1)
    slots = BoundedSemaphore(1)
    calls = []

    class Payload(BaseModel):
        query: str

    def handler(payload, _context):
        calls.append(payload.query)
        entered.set()
        release.wait(3)
        return payload

    manifest = SimpleNamespace(id="test.blocked",version="1.0.0",ref="test.blocked@1",
        input_model=Payload,output_model=Payload,timeout_ms=10,max_attempts=3,handler_callable=handler)
    registry = SimpleNamespace(resolve=lambda _ref: manifest)
    policies = SimpleNamespace(authorize=lambda *_args: [])
    monkeypatch.setattr(runtime,"_HANDLERS",pool)
    monkeypatch.setattr(runtime,"_HANDLER_SLOTS",slots)
    executor = runtime.SkillExecutor(registry, policies=policies)
    context = SkillContext(request_id="blocked",trace_id="blocked",deadline_at=datetime.now(timezone.utc)+timedelta(seconds=2))
    try:
        result = executor.execute("test.blocked@1", {"query":"one"}, context)
        assert entered.is_set()
        assert result.status == SkillStatus.RETRYABLE_ERROR
        assert not slots.acquire(blocking=False)
        executor.execute("test.blocked@1", {"query":"two"}, context)
        assert calls == ["one"]
    finally:
        release.set()
        pool.shutdown(wait=True)
    assert slots.acquire(blocking=False)
    slots.release()
