"""Shared public response contract for all mathematics tutor execution paths."""

from __future__ import annotations

from typing import Any, Literal
import uuid


ResponseType = Literal[
    "verified_answer",
    "guided_exercise",
    "clarification_required",
    "supported_refusal",
]


def normalize_response(payload: dict[str, Any], response_type: ResponseType) -> dict[str, Any]:
    """Normalize a producer dictionary and enforce public response guarantees."""
    allowed = {"verified_answer", "guided_exercise", "clarification_required", "supported_refusal"}
    if response_type not in allowed:
        raise ValueError(f"unsupported response type: {response_type}")
    result = {
        "answer": "",
        "trace_id": str(uuid.uuid4()),
        "intent": "",
        "knowledge_points": [],
        "sources": [],
        "validation_passed": False,
        "critic_report": {},
        "conversation_history": [],
        "conversation_summary": "",
        "exercise_state": None,
        "clarification": None,
        "metrics": {},
        "cached": False,
        **payload,
        "response_type": response_type,
    }
    if response_type == "verified_answer" and not result["validation_passed"]:
        raise ValueError("verified_answer requires validation_passed=True")
    if response_type == "guided_exercise" and (
        not result["validation_passed"]
        or not result["critic_report"].get("exercise_answer_hidden")
    ):
        raise ValueError("guided_exercise requires a verified hidden answer")
    return result


def _turns(query: str, history: list[dict[str, str]], answer: str) -> list[dict[str, str]]:
    return [*(history or []), {"role": "student", "content": query}, {"role": "tutor", "content": answer}]


def clarification_response(
    query: str,
    history: list[dict[str, str]],
    summary: str,
    missing: list[str],
    language: str = "zh",
) -> dict[str, Any]:
    if language == "en":
        answer = "I need one more detail before solving this: " + "; ".join(missing)
    else:
        answer = "为了准确解答，请补充：" + "；".join(missing)
    return normalize_response(
        {
            "answer": answer,
            "intent": "clarification",
            "sources": [],
            "validation_passed": True,
            "conversation_history": _turns(query, history, answer),
            "conversation_summary": summary,
            "clarification": {"missing": missing},
        },
        "clarification_required",
    )


def supported_refusal_response(
    query: str,
    history: list[dict[str, str]],
    summary: str,
    reason: str,
    language: str = "zh",
) -> dict[str, Any]:
    return normalize_response(
        {
            "answer": reason,
            "intent": "supported_refusal",
            "sources": [],
            "validation_passed": True,
            "conversation_history": _turns(query, history, reason),
            "conversation_summary": summary,
        },
        "supported_refusal",
    )
