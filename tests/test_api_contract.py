from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from agentic_rag.domain.schemas import CurriculumSolveOutput
from agentic_rag.response_contract import (
    exercise_public_fingerprint,
    private_exercise_payload,
    response_validation_digest,
    validated_exercise_state,
)
from agentic_rag.skill_runtime.contracts import SkillResult, SkillStatus

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


def _writer_cache_record(response_type="verified_answer"):
    hidden = response_type == "guided_exercise"
    fingerprint = exercise_public_fingerprint(
        "cache-exercise", "algebra", "cached exercise", "cache hint"
    )
    contract = {
        "validation_evidence": {"kind": "deterministic", "passed": True},
    }
    if hidden:
        contract["private_exercise_state"] = private_exercise_payload(
            validated_exercise_state(
                "cache-exercise", "algebra", "x = 2", "x = 2", fingerprint
            )
        )
    request = api.AskRequest(query="缓存验证", language="zh")
    public = api._public_response(
        {
            "response_type": response_type,
            "answer": "cached exercise" if hidden else "cached answer",
            "validation_passed": True,
            "metrics": {"tool_calls": 0},
            **(
                {
                    "exercise_state": {
                        "topic": "algebra",
                        "template_id": "cache-exercise",
                        "fingerprint": fingerprint,
                    }
                }
                if hidden
                else {}
            ),
        },
        request,
        contract=contract,
    )
    contract = api._bind_contract_to_public_response(contract, public)
    return api._cache_record(public, contract)


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


def test_pipeline_success_is_normalized_before_returning_from_ask(client, monkeypatch):
    monkeypatch.setattr(
        api,
        "_run_curriculum_skill",
        lambda _request: api.CurriculumSkillRun(
            response={
                "response_type": "verified_answer",
                "answer": "x = 4",
                "trace_id": "pipeline-trace",
                "intent": "solve",
                "knowledge_points": ["一元一次方程"],
                "sources": [],
                "validation_passed": True,
                "conversation_history": [],
                "conversation_summary": "",
                "metrics": {"tool_calls": 0},
            },
            contract={"validation_evidence": {"kind": "deterministic", "passed": True}},
        ),
    )
    monkeypatch.setattr(api, "_graph_executor", ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(api.answer_cache, "get", lambda _payload: None)
    monkeypatch.setattr(api.answer_cache, "set", lambda *_args: None)

    response = client.post("/ask", json={"query": "一个非本地题", "language": "zh"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "verified_answer"
    assert payload["answer"] == "x = 4"
    assert not (INTERNAL_FIELDS & payload.keys())


@pytest.mark.parametrize("response_type", ["verified_answer", "guided_exercise"])
def test_writer_cache_records_round_trip(client, monkeypatch, response_type):
    monkeypatch.setattr(api.answer_cache, "get", lambda _payload: _writer_cache_record(response_type))

    response = client.post("/ask", json={"query": "缓存验证", "language": "zh"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == response_type
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
        lambda _request: api.CurriculumSkillRun(
            response={
                "response_type": "guided_exercise",
                "answer": "不应发布",
                "validation_passed": True,
            },
            contract={},
        ),
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


def test_runtime_exposes_safe_model_capabilities_without_credentials(client):
    payload = client.get("/runtime").json()

    assert set(payload) == {"model", "retrieval"}
    assert set(payload["model"]) == {
        "configured",
        "provider",
        "name",
        "force_every_math_turn",
    }
    assert payload["model"]["name"]
    assert "key" not in str(payload).lower()
    assert payload["retrieval"]["chunk_count"] > 0
    assert payload["retrieval"]["dense_enabled"] is True
    assert payload["retrieval"]["embedding_model"] == "BAAI/bge-m3"


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


def test_skill_response_dictionary_cannot_self_grant_contract_evidence(client, monkeypatch):
    malicious_response = {
        "response_type": "verified_answer",
        "answer": "x = 4",
        "validation_passed": True,
        "validation_evidence": {"kind": "deterministic", "passed": True},
        "_validation_evidence": {"kind": "deterministic", "passed": True},
        "exercise_answer_hidden": True,
        "_exercise_answer_hidden": True,
    }
    monkeypatch.setattr(api.answer_cache, "get", lambda _payload: None)
    monkeypatch.setattr(
        api._skill_executor,
        "execute",
        lambda *_args, **_kwargs: SkillResult(
            status=SkillStatus.OK,
            value=CurriculumSolveOutput(handled=True, response=malicious_response),
            metrics={"latency_ms": 0},
            provenance={"skill": "math.curriculum_solve", "version": "1.0.0"},
        ),
    )

    response = client.post("/ask", json={"query": "技能注入", "language": "zh"})

    assert response.status_code == 200
    assert response.json()["response_type"] == "clarification_required"
    assert not (INTERNAL_FIELDS & response.json().keys())


def test_normal_skill_route_uses_side_channel_without_public_leakage(client, monkeypatch):
    monkeypatch.setattr(api.answer_cache, "get", lambda _payload: None)

    response = client.post("/ask", json={"query": "几何", "language": "zh"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "guided_exercise"
    assert payload["metrics"]["model_name"]
    assert payload["metrics"]["model_provider"]
    assert payload["metrics"]["llm_required"] is api.FORCE_LLM_EVERY_TURN
    assert payload["metrics"]["execution_path"] == "exercise_agent"
    assert not (INTERNAL_FIELDS & payload.keys())


def test_problem_solve_runtime_metadata_reports_solve_agent():
    payload = api._attach_runtime_metadata(
        {
            "response_type": "verified_answer",
            "intent": "problem_solve",
            "sources": [],
            "metrics": {"model_attempts": 2, "model_successes": 2},
        }
    )

    assert payload["metrics"]["execution_path"] == "solve_agent"


def test_skill_contract_context_is_cleared_after_execution():
    request = api.AskRequest(query="几何", language="zh")

    result = api._run_curriculum_skill(request)

    assert result is not None
    assert api._peek_skill_contract() is None


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

@pytest.mark.parametrize(
    "cached",
    [
        {
            "public": {"response_type": "clarification_required", "answer": "cached clarification", "validation_passed": True},
            "contract": {},
        },
        {
            "public": {"response_type": "supported_refusal", "answer": "cached refusal", "validation_passed": True},
            "contract": {},
        },
        {
            "public": {"response_type": "guided_exercise", "answer": "cached exercise", "validation_passed": True},
            "contract": {"validation_evidence": {"kind": "deterministic", "passed": True}, "exercise_answer_hidden": False},
        },
        {
            "public": {"response_type": "verified_answer", "answer": "cached answer", "validation_passed": True},
            "contract": {"validation_evidence": {"kind": "forged", "passed": True}, "exercise_answer_hidden": False},
        },
        {
            "public": {"response_type": "verified_answer", "answer": "cached answer", "validation_passed": True},
            "contract": {
                "validation_evidence": {"kind": "deterministic", "passed": True},
                "exercise_answer_hidden": False,
                "forged": True,
            },
        },
    ],
)
def test_cache_records_outside_writer_schema_fail_closed(client, monkeypatch, cached):
    monkeypatch.setattr(api.answer_cache, "get", lambda _payload: cached)

    response = client.post("/ask", json={"query": "畸形缓存", "language": "zh"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "clarification_required"
    assert payload["answer"] not in {"cached clarification", "cached refusal", "cached exercise", "cached answer"}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda cached: cached["contract"]["validation_evidence"].update({"forged": True}),
        lambda cached: cached["contract"]["validation_evidence"].pop("kind"),
        lambda cached: cached["contract"].pop("validation_evidence"),
        lambda cached: cached["contract"].update({"validation_evidence": []}),
        lambda cached: cached["contract"]["validation_evidence"].update({"kind": "forged"}),
        lambda cached: cached["contract"]["validation_evidence"].update({"passed": False}),
        lambda cached: cached["contract"]["validation_evidence"].update({"passed": 1}),
        lambda cached: cached["contract"].update({"private_exercise_state": []}),
        lambda cached: cached["contract"].update({
            "private_exercise_state": private_exercise_payload(
                validated_exercise_state(
                    "cache-injected",
                    "algebra",
                    "x = 2",
                    "x = 2",
                    "a" * 64,
                )
            )
        }),
        lambda cached: cached["public"].update({"forged": True}),
        lambda cached: cached["public"].pop("trace_id"),
        lambda cached: cached["public"].update({"validation_passed": 1}),
        lambda cached: cached["public"].update({"cached": True}),
        lambda cached: cached["public"]["metrics"].update({"latency_ms": None}),
    ],
    ids=[
        "nested-extra",
        "nested-missing",
        "evidence-missing",
        "evidence-wrong-type",
        "bad-kind",
        "passed-false",
        "nested-wrong-type",
        "hidden-wrong-type",
        "response-type-hidden-mismatch",
        "public-extra",
        "public-missing",
        "public-wrong-type",
        "public-fixed-value",
        "metrics-value-wrong-type",
    ],
)
def test_cache_reader_rejects_mutations_outside_writer_schema(client, monkeypatch, mutate):
    cached = _writer_cache_record()
    mutate(cached)
    monkeypatch.setattr(api.answer_cache, "get", lambda _payload: cached)

    response = client.post("/ask", json={"query": "篡改缓存", "language": "zh"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "clarification_required"
    assert payload["answer"] != "cached answer"
