"""Central policy for converting internal failures into safe teaching responses."""

from __future__ import annotations

from typing import Literal
import unicodedata
import uuid

from agentic_rag.completeness import analyze_completeness


FailureKind = Literal[
    "timeout",
    "runtime_error",
    "retrieval_empty",
    "critic_rejected",
    "expired_exercise",
    "cache_error",
]


def _generic_follow_up(query: str, language: str, failure_kind: FailureKind) -> tuple[str, list[str]]:
    normalized = unicodedata.normalize("NFKC", query or "").lower()
    if failure_kind == "expired_exercise":
        return (
            (
                "Choose algebra, geometry, or linear functions and I will create a fresh verified exercise."
                if language == "en"
                else "请选择代数、几何或一次函数，我会重新生成一道经过校验的新题。"
            ),
            ["a fresh exercise topic" if language == "en" else "新的练习主题"],
        )
    if any(marker in normalized for marker in ("证明", "全等", "prove", "congruent")):
        return (
            (
                "Please provide every known condition, the exact conclusion to prove, and the last step you are confident about."
                if language == "en"
                else "请补充全部已知条件、准确的待证结论，以及你目前能够确认的最后一步。"
            ),
            [
                "known conditions, proof target, or confirmed step"
                if language == "en"
                else "已知条件、待证结论或已确认步骤"
            ],
        )
    if any(marker in normalized for marker in ("公式", "定理", "知识点", "formula", "theorem", "topic")):
        return (
            (
                "Please name the grade, chapter, or exact concept you want to review."
                if language == "en"
                else "请说明年级、章节或想查询的具体知识点。"
            ),
            ["grade, chapter, or concept" if language == "en" else "年级、章节或知识点"],
        )
    return (
        (
            "Please send the complete problem and the step or conclusion you want checked."
            if language == "en"
            else "请发送完整题目，并指出希望我检查的步骤或结论。"
        ),
        [
            "the complete problem and step to check"
            if language == "en"
            else "完整题目和需要检查的步骤"
        ],
    )


def resolve_failure(
    query: str,
    language: Literal["zh", "en"],
    history: list[dict[str, str]],
    summary: str,
    failure_kind: FailureKind,
    issues: list[str],
    verified_partial: str | None = None,
) -> dict:
    completeness = analyze_completeness(query, language)
    response_type = "clarification_required"
    if completeness.status == "out_of_scope":
        response_type = "supported_refusal"
        follow_up = completeness.follow_up
        missing = completeness.missing
    elif completeness.status in {"missing_conditions", "requires_image"}:
        follow_up = completeness.follow_up
        missing = completeness.missing
    else:
        follow_up, missing = _generic_follow_up(query, language, failure_kind)

    partial = (verified_partial or "").strip()
    if partial:
        answer = (
            f"**Verified so far**\n{partial}\n\n**What I still need**\n{follow_up}"
            if language == "en"
            else f"**目前已确认**\n{partial}\n\n**还需要补充**\n{follow_up}"
        )
    else:
        answer = follow_up
    trace_id = str(uuid.uuid4())
    return {
        "response_type": response_type,
        "answer": answer,
        "trace_id": trace_id,
        "intent": "reliability_clarification",
        "knowledge_points": [],
        "sources": [],
        "validation_passed": True,
        "conversation_history": [
            *(history or []),
            {"role": "student", "content": query},
            {"role": "tutor", "content": answer},
        ],
        "conversation_summary": summary,
        "exercise_state": None,
        "clarification": {"missing": missing},
        "metrics": {
            "internal_failure_kind": failure_kind,
            "internal_issue_count": len(issues),
        },
        "cached": False,
    }
