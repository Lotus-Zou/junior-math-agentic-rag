import time

from agentic_rag.nodes import (
    analyze_completeness_node,
    clarification_response_node,
    no_evidence_response_node,
    validation_failure_response_node,
)


def base_state(**overrides):
    state = {
        "query": "",
        "response_language": "zh",
        "conversation_history": [],
        "conversation_summary": "",
        "trace_events": [],
        "step_count": 0,
        "deadline_at": time.time() + 30,
        "metrics": {},
        "validation_issues": [],
    }
    return state | overrides


def test_no_evidence_is_a_neutral_clarification():
    result = no_evidence_response_node(base_state(query="证明两三角形全等"))

    assert result["response_type"] == "clarification_required"
    assert result["internal_failure_kind"] == "retrieval_empty"
    assert "知识库没有召回" not in result["response"]
    assert "已知条件" in result["response"]


def test_rejected_draft_is_never_returned():
    marker = "未经验证的结论：两个三角形全等"
    result = validation_failure_response_node(
        base_state(
            query="证明两三角形全等",
            draft_response=marker,
            validation_issues=["缺少夹角条件"],
        )
    )

    assert result["response_type"] == "clarification_required"
    assert result["internal_failure_kind"] == "critic_rejected"
    assert marker not in result["response"]
    assert marker not in str(result.get("conversation_history", []))
    assert "夹角条件" not in result["response"]
    assert "已知条件" in result["response"]


def test_completeness_node_routes_missing_problem_without_model_call():
    result = analyze_completeness_node(base_state(query="这题怎么做"))

    assert result["completeness_status"] == "missing_conditions"
    assert result["needs_clarification"] is True
    assert result["missing_conditions"] == ["完整题干"]
    assert "完整题目" in result["follow_up_question"]


def test_completeness_node_routes_missing_diagram_relations():
    result = analyze_completeness_node(base_state(query="如图，在△ABC中求∠A"))

    assert result["completeness_status"] == "requires_image"
    assert result["needs_clarification"] is True
    assert result["missing_conditions"] == ["图形或图中已知关系"]


def test_complete_problem_continues_to_retrieval_path():
    result = analyze_completeness_node(base_state(query="解方程 2x+3=11"))

    assert result["completeness_status"] == "complete"
    assert result["needs_clarification"] is False
    assert result["missing_conditions"] == []


def test_clarification_uses_named_completeness_gap():
    result = clarification_response_node(
        base_state(
            query="如图，在△ABC中求∠A",
            completeness_status="requires_image",
            missing_conditions=["图形或图中已知关系"],
            follow_up_question="请上传图形，或写出图中标注的条件。",
        )
    )

    assert result["response_type"] == "clarification_required"
    assert result["clarification"]["missing"] == ["图形或图中已知关系"]
    assert "上传图形" in result["response"]

