import os
import uuid

import pytest

from agentic_rag import memory, task_queue


@pytest.fixture
def queue(monkeypatch):
    url = os.getenv("QUEUE_TEST_REDIS_URL")
    if not url:
        pytest.skip("set QUEUE_TEST_REDIS_URL to an isolated Redis test database")
    import redis

    client = redis.from_url(url, decode_responses=True)
    prefix = "queue-test:" + uuid.uuid4().hex
    monkeypatch.setattr(task_queue, "_redis", client)
    monkeypatch.setattr(task_queue, "QUEUE_KEY", prefix+":stream")
    monkeypatch.setattr(task_queue, "DEAD_LETTER_KEY", prefix+":dead")
    monkeypatch.setattr(task_queue, "_key", lambda task_id: prefix+":"+task_id)
    yield task_queue
    keys = list(client.scan_iter(prefix+"*"))
    if keys:
        client.delete(*keys)


def test_expired_worker_cannot_complete_or_fail_new_owners_task(queue):
    task_id = queue.enqueue("evaluation", {})
    old = queue.next_item(timeout=1)
    assert queue.claim(old)
    queue._redis.delete(queue._key(task_id)+":lease")
    new = dict(old)
    assert queue.claim(new)
    assert new["attempts"] == 2
    assert not queue.complete(task_id, {"owner": "stale"}, old)
    assert not queue.fail(task_id, "stale failure", old)
    assert queue.get_task(task_id)["status"] == "running"
    assert queue.renew_claim(new)
    assert queue.complete(task_id, {"owner": "new"}, new)
    assert queue.get_task(task_id)["result"] == {"owner": "new"}
    assert queue.queue_status()["depth"] == 0


def test_retry_attempts_survive_requeue_and_end_in_dead_letter(queue):
    task_id = queue.enqueue("unknown", {})
    for attempt in range(1, queue.TASK_MAX_ATTEMPTS+1):
        item = queue.next_item(timeout=1)
        assert queue.claim(item)
        assert item["attempts"] == attempt
        assert queue.fail(task_id, "injected", item) == (attempt < queue.TASK_MAX_ATTEMPTS)
    assert queue.get_task(task_id)["status"] == "failed"
    assert queue.queue_status()["depth"] == 0
    assert queue.queue_status()["dead_letter_depth"] == 1


def test_atomic_queue_capacity_includes_running_tasks_and_keeps_retries(queue, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(queue, "TASK_QUEUE_CAPACITY", 2)

    def submit(_):
        try:
            return queue.enqueue("evaluation", {})
        except queue.QueueCapacityExceeded:
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        submitted = list(executor.map(submit, range(24)))
    assert len([task for task in submitted if task]) == 2
    assert queue.queue_status()["depth"] == 2
    keys = list(queue._redis.scan_iter(queue._key("*")))
    assert sum(queue._redis.type(key) == "hash" for key in keys) == 2
    item = queue.next_item(timeout=1)
    assert queue.claim(item)
    with pytest.raises(queue.QueueCapacityExceeded):
        queue.enqueue("evaluation", {})
    assert queue.fail(item["task_id"], "retry at capacity", item)
    assert queue.queue_status()["depth"] == 2
    item = queue.next_item(timeout=1)
    assert queue.claim(item)
    assert queue.complete(item["task_id"], {}, item)
    assert queue.enqueue("evaluation", {})
    assert queue.queue_status()["depth"] == 2


def test_archive_effect_is_idempotent_even_when_vector_upsert_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "MYSQL_URL", "")
    monkeypatch.setattr(memory, "DB_PATH", str(tmp_path / "memory.sqlite"))

    class Collection:
        attempts = 0

        def upsert(self, **_kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("injected crash after SQL commit")

    collection = Collection()
    monkeypatch.setattr(memory, "_collection", lambda: collection)
    with pytest.raises(RuntimeError):
        memory.add_memory("lost a sign", idempotency_key="task-1")
    memory.add_memory("lost a sign", idempotency_key="task-1")
    with memory.get_db_connection() as connection:
        assert connection.execute("SELECT count(*) FROM memories").fetchone()[0] == 1
