"""Exercise the real production graph, renderer, API gate, and cache binding."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

import app as api
from agentic_rag import chains, skill_handlers
from agentic_rag.domain.schemas import (
    AnswerDraftOutput, AnswerRepairInput, CriticOutput, RetrievalOutput,
    CurriculumTutorInput,
)
from agentic_rag.graph import build_skill_pipeline_graph
from agentic_rag.reliability import resolve_failure


CHOICE_QUERY = (
    "计算 11+4 的结果。\nA. 14\nB. 15\nC. 16\nD. 17\n"
    "请解释解题思路，最后一行写答案字母，格式为：答案：A。"
)


@pytest.fixture
def twice_rejected_graph(monkeypatch):
    monkeypatch.setattr(skill_handlers, "retrieve_dense", lambda *_: RetrievalOutput())
    monkeypatch.setattr(skill_handlers, "retrieve_bm25", lambda *_: RetrievalOutput())
    monkeypatch.setattr(skill_handlers, "retrieve_graph", lambda *_: RetrievalOutput())
    monkeypatch.setattr(skill_handlers, "answer_generate", lambda *_: AnswerDraftOutput(answer="答案：A"))
    monkeypatch.setattr(skill_handlers, "answer_repair", lambda *_: AnswerDraftOutput(answer="答案：C"))
    monkeypatch.setattr(skill_handlers, "answer_critic", lambda *_: CriticOutput(
        passed=False, factual_faithfulness=False, math_logic_valid=False,
        issues=["arithmetic rejected"], critic={"validation_mode": "llm"},
    ))
    graph = build_skill_pipeline_graph()
    monkeypatch.setattr(api, "get_graph", lambda: graph)
    return graph


def test_final_judge_proof_survives_graph_renderer_api_and_cache(monkeypatch, twice_rejected_graph):
    calls = []

    def judge(payload):
        calls.append(payload)
        return {"is_complete": True, "math_logic_valid": True,
                "corrected_answer": "11+4=15，因此选择 B。\n答案：B", "issues": []}

    monkeypatch.setattr(chains, "get_final_answer_judge_chain", lambda: SimpleNamespace(invoke=judge))
    request = api.AskRequest(query=CHOICE_QUERY, request_id="proof-final-judge")
    run = api._run_curriculum_skill(request)
    assert run is not None
    assert len(calls) == 1
    assert run.contract["validation_evidence"] == {"kind": "independent_critic", "passed": True}
    public = api._public_response(run.response, request, contract=run.contract)
    assert public["response_type"] == "verified_answer"
    assert public["answer"].endswith("答案：B")
    assert "validation_evidence" not in public
    record = api._cache_record(public, api._bind_contract_to_public_response(run.contract, public))
    assert api._cached_response(record, request)["answer"] == public["answer"]
    with pytest.raises(api.ContractViolation):
        api._public_response({**run.response, "answer": "答案：D"}, request, contract=run.contract)


def test_final_judge_failure_cannot_publish_rejected_draft(monkeypatch, twice_rejected_graph):
    def judge(_):
        raise RuntimeError("judge unavailable")

    monkeypatch.setattr(chains, "get_final_answer_judge_chain", lambda: SimpleNamespace(invoke=judge))
    request = api.AskRequest(query=CHOICE_QUERY)
    run = api._run_curriculum_skill(request)
    assert run is not None
    public = api._public_response(run.response, request, contract=run.contract)
    assert public["response_type"] == "clarification_required"
    assert not api._cacheable_response(public, True)


def test_incomplete_final_judgment_cannot_create_verification_proof(monkeypatch, twice_rejected_graph):
    monkeypatch.setattr(chains, "get_final_answer_judge_chain", lambda: SimpleNamespace(
        invoke=lambda _: {"is_complete": False, "math_logic_valid": True,
                          "corrected_answer": "缺少一个已知条件，不能唯一确定选项。",
                          "issues": ["missing condition"]},
    ))
    request = api.AskRequest(query=CHOICE_QUERY)
    run = api._run_curriculum_skill(request)
    assert run is not None
    assert not run.contract
    assert api._public_response(run.response, request, contract=run.contract)["response_type"] == "clarification_required"


def test_audit_final_judge_captures_proof_for_api_gate(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from agentic_rag.response_contract import capture_skill_contract, consume_skill_contract
    from agentic_rag.skill_runtime.contracts import SkillContext

    monkeypatch.setattr(chains, "get_first_error_judge_chain", lambda: SimpleNamespace(
        invoke=lambda _: {"first_error_step": 2, "explanation": "2+2=4, not 5."},
    ))
    request = api.AskRequest(
        query="Audit this solution.\nStep 1. x = 3.\nStep 2. 2 + 2 = 5.\nWrite only 'First error: N' on the final line.",
        language="en",
    )
    context = SkillContext(request_id="audit-proof", trace_id="audit-proof",
                           deadline_at=datetime.now(timezone.utc)+timedelta(seconds=5))
    with capture_skill_contract() as capture:
        response = skill_handlers.audit_final_judge(
            AnswerRepairInput(query=request.query, answer="rejected", intent="error_analysis", language="en"), context,
        )
        contract = consume_skill_contract(capture)
    public = api._public_response(response.model_dump(mode="json"), request, contract=contract)
    assert public["answer"].endswith("First error: 2")
    assert public["response_type"] == "verified_answer"


def test_repair_allows_self_contained_derivations_without_fabricated_citations(monkeypatch):
    captured = []

    def invoke(_instance, messages):
        captured.extend(messages)
        return AIMessage(content="11+4=15。\n答案：B")

    monkeypatch.setattr(type(chains.generator_llm), "invoke", invoke)
    result = skill_handlers.answer_repair(
        AnswerRepairInput(query=CHOICE_QUERY, answer="答案：A", intent="multiple_choice"),
        SimpleNamespace(feature_flags={"llm_generate": True}),
    )
    assert result.answer.endswith("答案：B")
    assert "retrieved examples may be irrelevant" in captured[0].content
    assert "do not invent citations" in captured[0].content
    assert "Use only numbered evidence" not in captured[0].content


@pytest.mark.parametrize("kind", ["timeout", "runtime_error", "cache_error"])
@pytest.mark.parametrize("language", ["zh", "en"])
def test_internal_failure_does_not_blame_complete_input(kind, language):
    result = resolve_failure("计算 11+4", language, [], "", kind, ["private exception"])
    assert result["response_type"] == "clarification_required"
    assert "private exception" not in result["answer"]
    assert "请发送完整题目" not in result["answer"]
    assert "Please send the complete problem" not in result["answer"]
    assert "已保留" in result["answer"] if language == "zh" else "preserved" in result["answer"]


@pytest.mark.parametrize("flags,expected_calls", [
    ({"tutor_agent": True}, 0),
    ({"tutor_agent": True, "force_llm_every_turn": True}, 1),
    ({"tutor_agent": True, "enrich_verified_local_answers": True}, 1),
])
def test_verified_solver_explanation_does_not_require_optional_llm(monkeypatch, flags, expected_calls):
    from agentic_rag import tutor_agent
    from agentic_rag.domain.schemas import ResponseEnvelope

    baseline = ResponseEnvelope.model_validate(api.build_fast_response("解方程 2x+3=11", [], language="zh"))
    calls = []

    def enrich(**kwargs):
        calls.append(kwargs)
        return kwargs["baseline"]

    monkeypatch.setattr(tutor_agent, "enrich_curriculum_response", enrich)
    result = skill_handlers.curriculum_tutor(
        CurriculumTutorInput(query="解方程 2x+3=11", response=baseline),
        SimpleNamespace(feature_flags=flags),
    )
    assert len(calls) == expected_calls
    assert result.response.answer == baseline.answer
    assert result.response.validation_passed is True
