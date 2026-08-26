import pytest
from pydantic import ValidationError

from agentic_rag.response_contract import (
    clarification_response,
    normalize_response,
    supported_refusal_response,
)


def test_verified_answer_requires_validation():
    with pytest.raises(ValueError, match="verified_answer"):
        normalize_response({"answer": "x = 4", "validation_passed": False}, "verified_answer")


def test_public_response_types_are_restricted():
    with pytest.raises(ValueError):
        normalize_response({}, "draft")


def test_guided_exercise_requires_verified_hidden_answer():
    with pytest.raises(ValueError, match="guided_exercise"):
        normalize_response({"validation_passed": True}, "guided_exercise")


def test_clarification_names_missing_input_without_internal_copy():
    result = clarification_response("如图，求角A", [], "", ["图中已知角和点的位置关系"], "zh")
    assert result["response_type"] == "clarification_required"
    assert "图中已知角和点的位置关系" in result["answer"]
    assert "复杂推理服务" not in result["answer"]
    assert result["sources"] == []
    assert result["clarification"] == {"missing": ["图中已知角和点的位置关系"]}
    assert result["conversation_history"][-2:] == [
        {"role": "student", "content": "如图，求角A"},
        {"role": "tutor", "content": result["answer"]},
    ]


def test_supported_refusal_is_public_and_keeps_internal_state_out_of_answer():
    result = supported_refusal_response(
        "请完成超出范围的任务", [], "", "该请求不在初中数学辅导范围内", "zh"
    )
    assert result["response_type"] == "supported_refusal"
    assert result["answer"] == "该请求不在初中数学辅导范围内"
    assert result["sources"] == []
    assert result["validation_passed"] is True


def test_response_envelope_is_strict_and_exposes_contract_type():
    from agentic_rag.domain.schemas import ResponseEnvelope

    result = normalize_response(
        {"answer": "x = 4", "validation_passed": True, "validation_evidence": {"kind": "deterministic", "passed": True}}, "verified_answer"
    )
    envelope = ResponseEnvelope.model_validate(result)
    assert envelope.response_type == "verified_answer"
    with pytest.raises(ValidationError):
        ResponseEnvelope.model_validate({**result, "unexpected": True})



def test_verified_answer_requires_validation_evidence():
    for evidence in (None, {"kind": "model", "passed": True}, {"kind": "deterministic", "passed": False}):
        payload = {"answer": "x = 4", "validation_passed": True}
        if evidence is not None:
            payload["validation_evidence"] = evidence
        with pytest.raises(ValueError, match="verified_answer"):
            normalize_response(payload, "verified_answer")


def test_verified_answer_accepts_deterministic_or_independent_critic_evidence():
    for kind in ("deterministic", "independent_critic"):
        result = normalize_response(
            {
                "answer": "x = 4",
                "validation_passed": True,
                "validation_evidence": {"kind": kind, "passed": True},
            },
            "verified_answer",
        )
        assert result["response_type"] == "verified_answer"


def test_normalize_response_drops_internal_state_and_filters_metrics():
    result = normalize_response(
        {
            "answer": "x = 4",
            "validation_passed": True,
            "validation_evidence": {"kind": "deterministic", "passed": True},
            "critic_report": {"passed": True},
            "deadline_at": 123,
            "retrieval_trace": [{"query": "secret"}],
            "model_metadata": {"model": "private"},
            "metrics": {"tool_calls": 1, "latency_ms": 12, "timeout": True, "raw_tokens": 99},
        },
        "verified_answer",
    )
    assert "validation_evidence" not in result
    assert "critic_report" not in result
    assert "deadline_at" not in result
    assert "retrieval_trace" not in result
    assert "model_metadata" not in result
    assert result["metrics"] == {"tool_calls": 1, "latency_ms": 12}


def test_guided_exercise_requires_private_hidden_answer_signal():
    evidence = {"kind": "deterministic", "passed": True}
    for hidden in (None, False):
        payload = {"answer": "solve 3x=6", "validation_passed": True, "validation_evidence": evidence}
        if hidden is not None:
            payload["exercise_answer_hidden"] = hidden
        with pytest.raises(ValueError, match="guided_exercise"):
            normalize_response(payload, "guided_exercise")


def test_guided_exercise_accepts_private_hidden_answer_signal_without_leaking_it():
    result = normalize_response(
        {
            "answer": "Solve 3x = 6",
            "validation_passed": True,
            "validation_evidence": {"kind": "deterministic", "passed": True},
            "exercise_answer_hidden": True,
        },
        "guided_exercise",
    )
    assert result["response_type"] == "guided_exercise"
    assert "exercise_answer_hidden" not in result

