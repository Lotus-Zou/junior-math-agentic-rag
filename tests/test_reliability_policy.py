import pytest

from agentic_rag.reliability import resolve_failure


FORBIDDEN = (
    "复杂推理服务",
    "超时",
    "bad case",
    "模型",
    "重试",
    "trace",
    "provider",
    "internal detail",
)


@pytest.mark.parametrize(
    "kind",
    [
        "timeout",
        "runtime_error",
        "retrieval_empty",
        "critic_rejected",
        "expired_exercise",
        "cache_error",
    ],
)
def test_internal_failure_becomes_actionable_clarification(kind):
    result = resolve_failure(
        "证明这个几何结论",
        "zh",
        [],
        "",
        kind,
        ["internal detail", "provider secret"],
    )

    assert result["response_type"] == "clarification_required"
    assert all(token.lower() not in result["answer"].lower() for token in FORBIDDEN)
    assert result["metrics"]["internal_failure_kind"] == kind
    assert result["trace_id"]
    assert result["clarification"]["missing"]


def test_verified_partial_is_preserved_but_not_promoted():
    result = resolve_failure(
        "求解并证明",
        "zh",
        [],
        "",
        "critic_rejected",
        ["证明未验证"],
        verified_partial="已确定 x = 4；证明部分还缺少图形条件。",
    )

    assert result["response_type"] == "clarification_required"
    assert "已确定 x = 4" in result["answer"]
    assert "证明部分" in result["answer"]
    assert result["validation_passed"] is True


def test_named_completeness_gap_wins_over_generic_failure_copy():
    result = resolve_failure(
        "如图，在△ABC中求∠A",
        "zh",
        [],
        "",
        "retrieval_empty",
        ["no chunks"],
    )

    assert result["response_type"] == "clarification_required"
    assert "图中" in result["answer"]
    assert result["clarification"]["missing"] == ["图形或图中已知关系"]


def test_failure_policy_preserves_safe_conversation_shape():
    history = [{"role": "student", "content": "上一题"}]
    result = resolve_failure(
        "prove these triangles congruent",
        "en",
        history,
        "summary",
        "runtime_error",
        ["RuntimeError: key"],
    )

    assert result["conversation_history"][:-2] == history
    assert result["conversation_history"][-2]["role"] == "student"
    assert result["conversation_history"][-1]["role"] == "tutor"
    assert result["conversation_summary"] == "summary"
    assert "runtimeerror" not in result["answer"].lower()


def test_out_of_scope_completeness_becomes_supported_refusal():
    result = resolve_failure(
        "帮我写一首诗",
        "zh",
        [],
        "",
        "runtime_error",
        [],
    )

    assert result["response_type"] == "supported_refusal"
    assert "初中数学" in result["answer"]
