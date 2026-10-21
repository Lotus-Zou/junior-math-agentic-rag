"""Learning workspace boundaries, bounded task admission and honest report counts."""
from queue import Queue

import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

import app as api
from agentic_rag import domain_store, memory, task_queue


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "REQUIRE_USER_IDENTITY", False)
    monkeypatch.setattr(api, "AUTH_SHARED_SECRET", "")
    return TestClient(api.app)


@pytest.mark.parametrize(("exception", "status"), [
    (task_queue.QueueCapacityExceeded, 429),
    (task_queue.QueueUnavailable, 503),
])
def test_task_admission_returns_retryable_status_without_leaking_errors(client, monkeypatch, exception, status):
    def reject(*_args, **_kwargs):
        raise exception("private connection details")

    monkeypatch.setattr(api, "enqueue", reject)
    response = client.post("/tasks", json={"kind": "learning_report"})
    assert response.status_code == status
    assert response.headers["Retry-After"] == "5"
    assert response.headers["Cache-Control"] == "no-store"
    assert "private" not in response.text


def test_task_scope_comes_from_authenticated_headers(client, monkeypatch):
    captured = {}

    def enqueue(kind, payload, **scope):
        captured.update(scope)
        return "task-new"

    monkeypatch.setattr(api, "enqueue", enqueue)
    response = client.post("/tasks", json={"kind": "learning_report"},
                           headers={"X-User-ID": "student-a", "X-Session-ID": "session-a"})
    assert response.status_code == 202
    assert captured == {"user_id": "student-a", "session_id": "session-a"}


@pytest.mark.parametrize("field,value", [("user_id", "other-user"), ("session_id", "other-session")])
def test_other_owner_or_session_cannot_read_task(client, monkeypatch, field, value):
    task = {"user_id": "student-a", "session_id": "session-a", "status": "done", "result": {"private": True}}
    task[field] = value
    monkeypatch.setattr(api, "get_task", lambda _task_id: task)
    response = client.get("/tasks/task-new", headers={"X-User-ID": "student-a", "X-Session-ID": "session-a"})
    assert response.status_code == 404
    assert "private" not in response.text


def test_task_status_outage_is_not_a_not_found(client, monkeypatch):
    def unavailable(_task_id):
        raise task_queue.QueueUnavailable("offline")

    monkeypatch.setattr(api, "get_task", unavailable)
    assert client.get("/tasks/task-new").status_code == 503


def test_task_payload_size_is_bounded_in_bytes(client, monkeypatch):
    monkeypatch.setattr(api, "enqueue", lambda *_args, **_kwargs: pytest.fail("oversized task published"))
    assert client.post("/tasks", json={"kind": "evaluation", "payload": {"answer": "x" * 65537}}).status_code == 422
    assert client.post("/tasks", json={"kind": "evaluation", "payload": {"answer": "题" * 22000}}).status_code == 422


def test_local_queue_rejects_without_orphaning_task_and_releases_capacity(monkeypatch):
    monkeypatch.setattr(task_queue, "_redis", None)
    monkeypatch.setattr(task_queue, "_redis_required", False)
    monkeypatch.setattr(task_queue, "_local", Queue(maxsize=2))
    monkeypatch.setattr(task_queue, "_local_tasks", {})
    monkeypatch.setattr(task_queue, "TASK_QUEUE_CAPACITY", 1)
    task_id = task_queue.enqueue("learning_report", {})
    item = task_queue.next_item(timeout=1)
    assert task_queue.claim(item)
    with pytest.raises(task_queue.QueueCapacityExceeded):
        task_queue.enqueue("learning_report", {})
    assert list(task_queue._local_tasks) == [task_id]
    assert task_queue.complete(task_id, {}, item)
    assert task_queue.enqueue("learning_report", {})


def test_configured_redis_never_silently_falls_back_to_memory(monkeypatch):
    monkeypatch.setattr(task_queue, "_redis", None)
    monkeypatch.setattr(task_queue, "_redis_required", True)
    with pytest.raises(task_queue.QueueUnavailable):
        task_queue.enqueue("learning_report", {})
    with pytest.raises(task_queue.QueueUnavailable):
        task_queue.get_task("missing")


def test_redis_connection_errors_are_translated(monkeypatch):
    class Offline:
        def xgroup_create(self, *_args, **_kwargs):
            raise RedisConnectionError("secret host")

        def hgetall(self, *_args):
            raise RedisConnectionError("secret host")

    monkeypatch.setattr(task_queue, "_redis", Offline())
    with pytest.raises(task_queue.QueueUnavailable):
        task_queue.enqueue("learning_report", {})
    with pytest.raises(task_queue.QueueUnavailable):
        task_queue.get_task("missing")


@pytest.fixture
def domain(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "MYSQL_URL", "")
    monkeypatch.setattr(memory, "DB_PATH", str(tmp_path / "learning.sqlite"))
    monkeypatch.setattr(domain_store, "_mysql", lambda: None)
    monkeypatch.setattr(domain_store, "_schema_ready", False)
    domain_store.initialize_domain_schema()
    return domain_store


def test_report_total_is_not_truncated_to_recent_analysis_window(domain):
    user_id = domain.ensure_user("student-a")
    with memory.get_db_connection() as connection:
        connection.executemany(
            "INSERT INTO mistake_questions(user_id,session_key,question_text,knowledge_points,error_type) VALUES (?,?,?,?,?)",
            [(user_id, "session-a", str(index), '["algebra"]', "sign") for index in range(205)],
        )
        connection.execute("INSERT INTO mistake_questions(user_id,session_key,question_text) VALUES (?,?,?)", (user_id, "session-b", "private"))
    report = domain.learning_report("student-a", "session-a")
    assert report["mistake_count"] == 205
    assert report["analyzed_count"] == report["analysis_limit"] == 200
    assert report["top_knowledge_points"] == [("algebra", 200)]
    rows = domain.list_mistakes("student-a", "session-a", 2)
    assert [row["question_text"] for row in rows] == ["204", "203"]
    assert domain.learning_report("student-b", "session-a")["mistake_count"] == 0
    assert domain.list_mistakes("student-b", "session-a") == []


def test_empty_report_has_no_invented_learning_results(domain):
    report = domain.learning_report("new-student", "new-session")
    assert report["mistake_count"] == report["analyzed_count"] == 0
    assert report["top_error_types"] == report["top_knowledge_points"] == []
