"""Shared public response contract for all mathematics tutor execution paths."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal
import uuid


@dataclass
class SkillContractCapture:
    contract: dict[str, Any] | None = None


_skill_contract_capture: ContextVar[SkillContractCapture | None] = ContextVar(
    "skill_contract_capture", default=None
)


@contextmanager
def capture_skill_contract():
    capture = SkillContractCapture()
    token = _skill_contract_capture.set(capture)
    try:
        yield capture
    finally:
        capture.contract = None
        _skill_contract_capture.reset(token)


def consume_skill_contract(capture: SkillContractCapture) -> dict[str, Any]:
    contract, capture.contract = capture.contract, None
    return dict(contract or {})


def peek_skill_contract() -> dict[str, Any] | None:
    capture = _skill_contract_capture.get()
    return None if capture is None or capture.contract is None else dict(capture.contract)


def _store_skill_contract(evidence: Any, hidden: bool) -> None:
    capture = _skill_contract_capture.get()
    if capture is None:
        return
    if (
        isinstance(evidence, dict)
        and evidence.get("kind") in {"deterministic", "independent_critic"}
        and evidence.get("passed") is True
    ):
        capture.contract = {
            "validation_evidence": {"kind": evidence["kind"], "passed": True},
            "exercise_answer_hidden": hidden,
        }


def capture_trusted_skill_response_contract(payload: dict[str, Any]) -> None:
    """Store a repository-owned fast-path contract outside its public response."""
    if _skill_contract_capture.get() is None or peek_skill_contract() is not None:
        return
    critic = payload.get("critic_report")
    if (
        isinstance(critic, dict)
        and payload.get("validation_passed") is True
        and critic.get("is_valid") is True
    ):
        kind = "deterministic" if str(critic.get("validation_mode", "")).startswith("local_") else "independent_critic"
        _store_skill_contract(
            {"kind": kind, "passed": True},
            critic.get("exercise_answer_hidden") is True,
        )


ResponseType = Literal[
    "verified_answer",
    "guided_exercise",
    "clarification_required",
    "supported_refusal",
]


def normalize_response(payload: dict[str, Any], response_type: ResponseType) -> dict[str, Any]:
    """Project producer state onto the public contract and enforce validation gates."""
    allowed = {"verified_answer", "guided_exercise", "clarification_required", "supported_refusal"}
    if response_type not in allowed:
        raise ValueError(f"unsupported response type: {response_type}")

    evidence = payload.get("validation_evidence")
    evidence_ok = (
        isinstance(evidence, dict)
        and evidence.get("kind") in {"deterministic", "independent_critic"}
        and evidence.get("passed") is True
    )
    if response_type in {"verified_answer", "guided_exercise"} and (
        payload.get("validation_passed") is not True or not evidence_ok or (response_type == "guided_exercise" and payload.get("exercise_answer_hidden") is not True)
    ):
        raise ValueError(f"{response_type} requires explicit passing validation evidence")

    public_fields = (
        "answer", "trace_id", "intent", "knowledge_points", "sources",
        "validation_passed", "conversation_history", "conversation_summary",
        "exercise_state", "clarification", "cached",
    )
    result = {field: payload[field] for field in public_fields if field in payload}
    result.setdefault("answer", "")
    result.setdefault("trace_id", str(uuid.uuid4()))
    result.setdefault("intent", "")
    result.setdefault("knowledge_points", [])
    result.setdefault("sources", [])
    result.setdefault("validation_passed", False)
    result.setdefault("conversation_history", [])
    result.setdefault("conversation_summary", "")
    result.setdefault("exercise_state", None)
    result.setdefault("clarification", None)
    result.setdefault("cached", False)
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        result["metrics"] = {
            key: metrics[key] for key in ("tool_calls", "latency_ms") if key in metrics
        }
    else:
        result["metrics"] = {}
    result["response_type"] = response_type
    if response_type in {"verified_answer", "guided_exercise"}:
        _store_skill_contract(evidence, payload.get("exercise_answer_hidden") is True)
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

