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
        {"answer": "x = 4", "validation_passed": True}, "verified_answer"
    )
    envelope = ResponseEnvelope.model_validate(result)
    assert envelope.response_type == "verified_answer"
    with pytest.raises(ValidationError):
        ResponseEnvelope.model_validate({**result, "unexpected": True})


