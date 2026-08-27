from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import is_dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import agentic_rag.fast_path as fast_path
import app as api
from agentic_rag.domain.schemas import CurriculumSolveOutput, ResponseEnvelope
from agentic_rag.fast_path import build_fast_response
from agentic_rag.nodes import no_evidence_response_node, prepare_retry_node
from agentic_rag.response_contract import normalize_response
from agentic_rag.skill_runtime.contracts import SkillContext
from agentic_rag.skill_runtime.executor import SkillExecutor
from agentic_rag.skill_runtime.pipeline import PipelineExecutor, PipelineLoader
from agentic_rag.skill_runtime.registry import get_default_registry


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_RESPONSE_TYPES = {
    "verified_answer",
    "guided_exercise",
    "clarification_required",
    "supported_refusal",
}


@pytest.fixture
def client():
    with TestClient(api.app, raise_server_exceptions=False) as test_client:
        yield test_client


def _context(name: str = "phase-1-final") -> SkillContext:
    return SkillContext(
        request_id=name,
        trace_id=name,
        deadline_at=datetime.now(timezone.utc) + timedelta(seconds=8),
    )


def _tamper_hidden_answer(template, bad_answer: str):
    if is_dataclass(template):
        return replace(template, hidden_answer=bad_answer)
    values = list(template)
    if len(values) == 2:
        values[0] = "0x = 1"
    else:
        values[-1] = bad_answer
    return tuple(values)


def test_graph_contract_rejects_unknown_mode_and_unbound_final_response():
    old_answer = "validated old draft"
    stale_critic = {
        "is_valid": True,
        "validation_mode": "llm",
        "deterministic": {"passed": True},
        "validated_response_sha256": hashlib.sha256(old_answer.encode("utf-8")).hexdigest(),
    }
    assert api._graph_contract(
        {
            "response": "new no-evidence response",
            "draft_response": "",
            "response_type": "clarification_required",
            "validation_passed": True,
            "critic_report": stale_critic,
        }
    ) == {}
    assert api._graph_contract(
        {
            "response": old_answer,
            "draft_response": old_answer,
            "response_type": "verified_answer",
            "validation_passed": True,
            "critic_report": {**stale_critic, "validation_mode": "future_magic"},
        }
    ) == {}


def test_retry_and_terminal_degrade_clear_stale_graph_validation_state():
    state = {
        "query": "一道未覆盖题",
        "draft_response": "old draft",
        "response": "old draft",
        "response_type": "verified_answer",
        "validation_passed": False,
        "critic_report": {
            "is_valid": True,
            "validation_mode": "llm",
            "validated_response_sha256": hashlib.sha256(b"old draft").hexdigest(),
        },
        "correction_attempts": 0,
        "trace_events": [],
    }

    retry = prepare_retry_node(state)
    retried = {**state, **retry}
    assert retried["response"] == ""
    assert retried["critic_report"] == {}
    assert retried["validation_passed"] is False

    degraded = {**retried, **no_evidence_response_node(retried)}
    assert degraded["response_type"] == "clarification_required"
    assert degraded["critic_report"] == {}
    assert api._graph_contract(degraded) == {}


def test_stale_graph_evidence_is_not_published_or_cached(client, monkeypatch):
    class Graph:
        def invoke(self, *_args, **_kwargs):
            old_answer = "first critic-approved draft"
            return {
                "response": "教材依据不足，请补充完整题目。",
                "draft_response": "",
                "response_type": "clarification_required",
                "trace_id": "stale-graph",
                "validation_passed": True,
                "critic_report": {
                    "is_valid": True,
                    "validation_mode": "llm",
                    "deterministic": {"passed": True},
                    "validated_response_sha256": hashlib.sha256(old_answer.encode("utf-8")).hexdigest(),
                },
                "conversation_history": [],
                "conversation_summary": "",
                "metrics": {"tool_calls": 2},
            }

    cache_writes = []
    monkeypatch.setattr(api, "_run_curriculum_skill", lambda _request: None)
    monkeypatch.setattr(api, "get_graph", lambda: Graph())
    monkeypatch.setattr(api, "_graph_executor", ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(api.answer_cache, "get", lambda _payload: None)
    monkeypatch.setattr(api.answer_cache, "set", lambda *args: cache_writes.append(args))

    response = client.post("/ask", json={"query": "一道未覆盖题", "language": "zh"})

    assert response.status_code == 200
    assert response.json()["response_type"] == "clarification_required"
    assert cache_writes == []


@pytest.mark.parametrize(
    ("query", "history", "template_name"),
    [
        (
            "再出一个类似的题",
            [
                {"role": "student", "content": "解方程 2x+3=11"},
                {"role": "tutor", "content": "x = 4"},
            ],
            "ZH_EXERCISES",
        ),
        ("出一个几何题我做做", [], "ZH_GEOMETRY_EXERCISES"),
    ],
)
def test_legacy_guided_routes_reject_tampered_hidden_answers(
    monkeypatch, query, history, template_name
):
    templates = getattr(fast_path, template_name)
    monkeypatch.setattr(
        fast_path,
        template_name,
        (_tamper_hidden_answer(templates[0], "tampered hidden answer"),),
    )

    result = build_fast_response(query, history, language="zh")

    assert result["response_type"] == "clarification_required"
    assert "tampered hidden answer" not in str(result)
    assert "local_template_validation_failed" not in str(result)


def test_legacy_equation_route_does_not_publish_a_tampered_solver_answer(monkeypatch):
    monkeypatch.setattr(
        fast_path,
        "deterministic_equation_answer",
        lambda *_args, **_kwargs: "分步过程：x = 999。自检：正确。",
    )

    result = build_fast_response("解方程 2x+3=11", [], language="zh")

    assert result["response_type"] == "clarification_required"
    assert "x = 999" not in result["answer"]


@pytest.mark.parametrize(
    ("query", "history"),
    [
        (
            "再出一个类似的题",
            [
                {"role": "student", "content": "解方程 2x+3=11"},
                {"role": "tutor", "content": "x = 4"},
            ],
        ),
        ("出一个几何题我做做", []),
    ],
)
def test_guided_skill_contract_keeps_validated_hidden_answer_private(query, history):
    request = api.AskRequest(query=query, language="zh", conversation_history=history)

    run = api._run_curriculum_skill(request)

    assert run is not None
    private_state = run.contract["private_exercise_state"]
    assert private_state["hidden_answer"]
    assert private_state["validation_digest"]
    assert "hidden_answer" not in str(run.response)
    assert run.response["response_type"] == "guided_exercise"


def test_curriculum_skill_and_renderer_use_typed_public_envelopes():
    with pytest.raises(ValidationError):
        CurriculumSolveOutput(handled=True, response={"answer": "untyped"})

    registry = get_default_registry()
    curriculum = registry.resolve("math.curriculum_solve@1").output_model.model_json_schema()
    response_schema = curriculum["properties"]["response"]
    assert "additionalProperties" not in str(response_schema)
    assert "ResponseEnvelope" in str(response_schema)

    pipeline = PipelineLoader(registry).load(ROOT / "agentic_rag" / "pipelines" / "correction.yaml")
    runner = PipelineExecutor(SkillExecutor(registry))
    verified = runner.run(pipeline, {"query": "解方程 2x+3=11", "language": "zh"}, _context("typed-verified"))
    guided = runner.run(pipeline, {"query": "几何", "language": "zh"}, _context("typed-guided"))
    assert verified["curriculum_solve"].response.response_type == "verified_answer"
    assert guided["curriculum_solve"].response.response_type == "guided_exercise"
    assert verified["response_render"].response_type == "verified_answer"
    assert guided["response_render"].response_type == "guided_exercise"


@pytest.mark.parametrize("command", ["再来一道", "难一点", "简单一点"])
def test_recent_exercise_topic_wins_across_topic_history(command):
    algebra = build_fast_response("代数", [], language="zh")
    geometry = build_fast_response("几何", algebra["conversation_history"], language="zh")

    result = build_fast_response(command, geometry["conversation_history"], language="zh")

    assert result["response_type"] == "guided_exercise"
    assert result["exercise_state"]["topic"] == "geometry"


def test_complete_geometry_problem_is_not_routed_as_generation_command():
    result = build_fast_response(
        "出一个几何题：在直角三角形中，两直角边长为 3 和 4，求斜边。",
        [],
        language="zh",
    )

    assert result["response_type"] == "verified_answer"
    assert result["intent"] != "geometry_exercise"
    assert "c = 5" in result["answer"]


def test_nested_public_projection_is_recursive_and_keeps_supported_metadata():
    result = normalize_response(
        {
            "answer": "x = 4",
            "validation_passed": True,
            "validation_evidence": {"kind": "deterministic", "passed": True},
            "sources": [
                {
                    "chunk_id": "chunk-1",
                    "source": "textbook.md",
                    "chapter": "代数",
                    "rank": 1,
                    "metadata": {"model": "private"},
                    "critic": {"passed": True},
                }
            ],
            "exercise_state": {
                "topic": "algebra",
                "difficulty_delta": 1,
                "hidden_answer": "x = 4",
                "retrieval": {"query": "private"},
            },
            "clarification": {
                "missing": ["完整题干"],
                "model_reason": "private",
            },
        },
        "verified_answer",
    )

    assert result["sources"] == [
        {"chunk_id": "chunk-1", "source": "textbook.md", "chapter": "代数", "rank": 1}
    ]
    assert result["exercise_state"] == {"topic": "algebra", "difficulty_delta": 1}
    assert result["clarification"] == {"missing": ["完整题干"]}
    ResponseEnvelope.model_validate(result)
    with pytest.raises(ValidationError):
        ResponseEnvelope.model_validate(
            {
                **result,
                "sources": [{**result["sources"][0], "critic": {"passed": True}}],
            }
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda record: record["public"]["sources"][0].update({"critic": {"passed": True}}),
        lambda record: record["public"]["exercise_state"].update({"hidden_answer": "x = 4"}),
        lambda record: record["public"]["clarification"].update({"retrieval": {"query": "private"}}),
    ],
)
def test_cache_rejects_nested_public_mutations(client, monkeypatch, mutation):
    request = api.AskRequest(query="nested cache", language="zh")
    contract = {
        "validation_evidence": {"kind": "deterministic", "passed": True},
    }
    public = api._public_response(
        {
            "response_type": "verified_answer",
            "answer": "x = 4",
            "validation_passed": True,
            "sources": [{"chunk_id": "c1", "source": "book", "chapter": "代数", "rank": 1}],
            "exercise_state": {"topic": "algebra", "difficulty_delta": 0},
            "clarification": {"missing": []},
        },
        request,
        contract=contract,
    )
    record = api._cache_record(public, contract)
    mutation(record)
    monkeypatch.setattr(api.answer_cache, "get", lambda _payload: record)

    response = client.post("/ask", json={"query": "nested cache", "language": "zh"})

    assert response.status_code == 200
    assert response.json()["response_type"] == "clarification_required"
    assert response.json()["answer"] != "x = 4"


def test_metrics_are_disabled_without_token_and_authorized_with_bearer(client, monkeypatch):
    monkeypatch.setattr(api, "OPERATIONS_METRICS_TOKEN", "")
    disabled = client.get("/metrics")
    assert disabled.status_code == 404

    monkeypatch.setattr(api, "OPERATIONS_METRICS_TOKEN", "operations-secret")
    unauthorized = client.get("/metrics")
    wrong = client.get("/metrics", headers={"Authorization": "Bearer wrong"})
    authorized = client.get(
        "/metrics", headers={"Authorization": "Bearer operations-secret"}
    )
    assert unauthorized.status_code == 404
    assert wrong.status_code == 404
    assert "operations-secret" not in unauthorized.text + wrong.text
    assert authorized.status_code == 200
    assert "text/plain" in authorized.headers["content-type"]


def test_first_paint_copy_contains_only_student_facing_language():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8").lower()
    for forbidden in ("critic", "model", "retrieval", "timeout", "检索", "模型", "超时"):
        assert forbidden not in html
