from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

import app as api
from app import app


PUBLIC_RESPONSE_TYPES = {
    "verified_answer",
    "guided_exercise",
    "clarification_required",
    "supported_refusal",
}
INTERNAL_FIELDS = {
    "critic_report",
    "validation_evidence",
    "exercise_answer_hidden",
    "hidden_answer",
    "model_metadata",
    "retrieval_trace",
}


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.mark.parametrize(
    ("query", "language", "response_type"),
    [
        ("几何", "zh", "guided_exercise"),
        ("代数", "zh", "guided_exercise"),
        ("一次函数", "zh", "guided_exercise"),
        ("难一点", "zh", "clarification_required"),
        ("换个问题", "zh", "clarification_required"),
        ("reset", "en", "clarification_required"),
    ],
)
def test_short_commands_return_only_public_api_contract(client, query, language, response_type):
    response = client.post("/ask", json={"query": query, "language": language})

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == response_type
    assert payload["response_type"] in PUBLIC_RESPONSE_TYPES
    assert payload["metrics"]["tool_calls"] == 0
    assert not (INTERNAL_FIELDS & payload.keys())


def test_geometry_never_reaches_complex_reasoning(client):
    response = client.post("/ask", json={"query": "几何", "language": "zh"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "guided_exercise"
    assert payload["metrics"]["tool_calls"] == 0
    assert "复杂推理服务" not in payload["answer"]
    assert not (INTERNAL_FIELDS & payload.keys())


def test_new_question_clears_previous_state(client):
    response = client.post(
        "/ask",
        json={
            "query": "换个问题",
            "language": "zh",
            "conversation_summary": "旧摘要",
            "conversation_history": [{"role": "student", "content": "解方程 2x+3=11"}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "clarification_required"
    assert payload["conversation_summary"] == ""
    assert "2x+3=11" not in str(payload["conversation_history"])


def test_graph_success_is_normalized_before_returning_from_ask(client, monkeypatch):
    class Graph:
        def invoke(self, *_args, **_kwargs):
            return {
                "response": "x = 4",
                "trace_id": "graph-trace",
                "intent": "solve",
                "knowledge_points": ["一元一次方程"],
                "documents": [],
                "validation_passed": True,
                "critic_report": {"is_valid": True, "validation_mode": "local_sympy"},
                "conversation_history": [],
                "conversation_summary": "",
                "metrics": {"tool_calls": 0},
            }

    monkeypatch.setattr(api, "_run_curriculum_skill", lambda _request: None)
    monkeypatch.setattr(api, "get_graph", lambda: Graph())
    monkeypatch.setattr(api, "_graph_executor", ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(api.answer_cache, "get", lambda _payload: None)
    monkeypatch.setattr(api.answer_cache, "set", lambda *_args: None)

    response = client.post("/ask", json={"query": "一个非本地题", "language": "zh"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "verified_answer"
    assert payload["answer"] == "x = 4"
    assert not (INTERNAL_FIELDS & payload.keys())


def test_verified_cache_hit_requires_stored_validation_evidence(client, monkeypatch):
    monkeypatch.setattr(
        api.answer_cache,
        "get",
        lambda _payload: {
            "public": {
                "response_type": "verified_answer",
                "answer": "x = 4",
                "validation_passed": True,
                "metrics": {"tool_calls": 0},
            },
            "contract": {"validation_evidence": {"kind": "deterministic", "passed": True}},
        },
    )

    response = client.post("/ask", json={"query": "缓存验证", "language": "zh"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "verified_answer"
    assert payload["cached"] is True
    assert not (INTERNAL_FIELDS & payload.keys())


@pytest.mark.parametrize(
    "cached",
    [
        {"response_type": "verified_answer", "answer": "x = 4", "validation_passed": True},
        {
            "public": {"response_type": "verified_answer", "answer": "x = 4", "validation_passed": True},
            "contract": {"validation_evidence": {"kind": "untrusted", "passed": True}},
        },
    ],
)
def test_invalid_cached_contract_fails_closed_with_trace(client, monkeypatch, cached):
    calls = []
    monkeypatch.setattr(api.answer_cache, "get", lambda _payload: cached)
    monkeypatch.setattr(api, "_record_failure", lambda *args: calls.append(args))

    response = client.post("/ask", json={"query": "缓存异常", "language": "zh"})

    assert response.status_code == 200
    assert response.json()["response_type"] == "clarification_required"
    assert calls and calls[0][2] == "contract_error"


def test_invalid_skill_contract_and_skill_exception_fail_closed(client, monkeypatch):
    calls = []
    monkeypatch.setattr(api.answer_cache, "get", lambda _payload: None)
    monkeypatch.setattr(api, "_record_failure", lambda *args: calls.append(args))

    monkeypatch.setattr(
        api,
        "_run_curriculum_skill",
        lambda _request: {
            "response_type": "guided_exercise",
            "answer": "不应发布",
            "validation_passed": True,
        },
    )
    invalid = client.post("/ask", json={"query": "坏技能", "language": "zh"})

    monkeypatch.setattr(api, "_run_curriculum_skill", lambda _request: (_ for _ in ()).throw(RuntimeError("producer failed")))
    crashed = client.post("/ask", json={"query": "技能异常", "language": "zh"})

    assert invalid.status_code == crashed.status_code == 200
    assert invalid.json()["response_type"] == crashed.json()["response_type"] == "clarification_required"
    assert [call[2] for call in calls] == ["contract_error", "runtime_error"]


def test_health_and_ready_are_minimal_public_statuses(client):
    health = client.get("/health").json()
    ready = client.get("/ready").json()

    assert health == {"status": "ok"}
    assert set(ready) == {"status"}
    public_text = f"{health} {ready}".lower()
    for internal in ("timeout", "agent", "vector", "model", "configured", "index"):
        assert internal not in public_text


def test_public_cache_evidence_cannot_replace_contract_evidence(client, monkeypatch):
    monkeypatch.setattr(
        api.answer_cache,
        "get",
        lambda _payload: {
            "public": {
                "response_type": "verified_answer",
                "answer": "x = 4",
                "validation_passed": True,
                "validation_evidence": {"kind": "deterministic", "passed": True},
            },
            "contract": {},
        },
    )

    response = client.post("/ask", json={"query": "缓存注入", "language": "zh"})

    assert response.status_code == 200
    assert response.json()["response_type"] == "clarification_required"
    assert "validation_evidence" not in response.json()


def test_public_skill_evidence_cannot_replace_private_contract(client, monkeypatch):
    monkeypatch.setattr(api.answer_cache, "get", lambda _payload: None)
    monkeypatch.setattr(
        api,
        "_run_curriculum_skill",
        lambda _request: {
            "response_type": "verified_answer",
            "answer": "x = 4",
            "validation_passed": True,
            "validation_evidence": {"kind": "deterministic", "passed": True},
        },
    )

    response = client.post("/ask", json={"query": "技能注入", "language": "zh"})

    assert response.status_code == 200
    assert response.json()["response_type"] == "clarification_required"
    assert not (INTERNAL_FIELDS & response.json().keys())


def test_diagnostic_failures_do_not_break_fail_closed_response(client, monkeypatch):
    monkeypatch.setattr(api.answer_cache, "get", lambda _payload: {"invalid": "cache"})
    monkeypatch.setattr(api, "persist_trace", lambda _state: (_ for _ in ()).throw(OSError("trace store unavailable")))
    monkeypatch.setattr(api, "observe_state", lambda *_args: (_ for _ in ()).throw(RuntimeError("metrics unavailable")))

    response = client.post("/ask", json={"query": "诊断故障", "language": "zh"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "clarification_required"
    public_text = payload["answer"].lower()
    for internal in ("traceback", "oserror", "runtimeerror", "timeout", "model error"):
        assert internal not in public_text


def test_ready_reports_degraded_without_leaking_dependency_details(client, monkeypatch):
    monkeypatch.setattr(api, "_readiness_checks", lambda: {"static_ui": False})

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "degraded"}