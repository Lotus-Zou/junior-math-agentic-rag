"""Shared public response contract for all mathematics tutor execution paths."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import hmac
import json
from typing import Any, Literal
import uuid

from pydantic import BaseModel

from agentic_rag.domain.schemas import (
    PublicClarification,
    PublicConversationTurn,
    PublicExerciseState,
    PublicMetrics,
    PublicSource,
    ResponseEnvelope,
)


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


@dataclass(frozen=True)
class ValidatedExerciseState:
    """Private proof that a hidden answer matched an independent local solver."""

    template_id: str
    topic: str
    hidden_answer: str
    validation_digest: str


def _exercise_digest(template_id: str, topic: str, hidden_answer: str) -> str:
    payload = json.dumps(
        [template_id, topic, hidden_answer], ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validated_exercise_state(
    template_id: str,
    topic: str,
    hidden_answer: str,
    solved_answer: str,
) -> ValidatedExerciseState | None:
    """Create private exercise state only when a deterministic solver agrees."""
    values = (template_id.strip(), topic.strip(), hidden_answer.strip(), solved_answer.strip())
    if not all(values) or not hmac.compare_digest(
        values[2].encode("utf-8"), values[3].encode("utf-8")
    ):
        return None
    return ValidatedExerciseState(
        template_id=values[0],
        topic=values[1],
        hidden_answer=values[2],
        validation_digest=_exercise_digest(values[0], values[1], values[2]),
    )


def restore_validated_exercise_state(candidate: Any) -> ValidatedExerciseState | None:
    if isinstance(candidate, ValidatedExerciseState):
        state = candidate
    elif isinstance(candidate, dict) and set(candidate) == {
        "template_id", "topic", "hidden_answer", "validation_digest"
    }:
        try:
            state = ValidatedExerciseState(**candidate)
        except TypeError:
            return None
    else:
        return None
    expected = _exercise_digest(state.template_id, state.topic, state.hidden_answer)
    if not hmac.compare_digest(state.validation_digest, expected):
        return None
    return state


def private_exercise_payload(state: ValidatedExerciseState) -> dict[str, str]:
    return {
        "template_id": state.template_id,
        "topic": state.topic,
        "hidden_answer": state.hidden_answer,
        "validation_digest": state.validation_digest,
    }


def response_validation_digest(answer: str) -> str:
    return hashlib.sha256((answer or "").encode("utf-8")).hexdigest()


def _store_skill_contract(
    evidence: Any, private_exercise: ValidatedExerciseState | None
) -> None:
    capture = _skill_contract_capture.get()
    if capture is None:
        return
    if (
        isinstance(evidence, dict)
        and evidence.get("kind") in {"deterministic", "independent_critic"}
        and evidence.get("passed") is True
    ):
        contract = {
            "validation_evidence": {"kind": evidence["kind"], "passed": True},
        }
        if private_exercise is not None:
            contract["private_exercise_state"] = private_exercise_payload(private_exercise)
        capture.contract = contract


ResponseType = Literal[
    "verified_answer",
    "guided_exercise",
    "clarification_required",
    "supported_refusal",
]


def _project_model(model: type[BaseModel], value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{model.__name__} must be an object")
    projected = {field: value[field] for field in fields if field in value}
    return model.model_validate(projected).model_dump(mode="json", exclude_unset=True)


def _project_sources(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("sources must be a list")
    return [
        _project_model(PublicSource, item, ("chunk_id", "source", "chapter", "rank"))
        for item in value
    ]


def _project_turns(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("conversation_history must be a list")
    return [
        _project_model(PublicConversationTurn, item, ("role", "content"))
        for item in value
    ]


def normalize_response(
    payload: dict[str, Any],
    response_type: ResponseType,
    *,
    private_exercise: ValidatedExerciseState | None = None,
) -> dict[str, Any]:
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
    private_exercise = restore_validated_exercise_state(private_exercise)
    if response_type in {"verified_answer", "guided_exercise"} and (
        payload.get("validation_passed") is not True
        or not evidence_ok
        or (response_type == "guided_exercise" and private_exercise is None)
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
    result["sources"] = _project_sources(result.get("sources", []))
    result.setdefault("validation_passed", False)
    result["conversation_history"] = _project_turns(result.get("conversation_history", []))
    result.setdefault("conversation_summary", "")
    if result.get("exercise_state") is not None:
        result["exercise_state"] = _project_model(
            PublicExerciseState,
            result["exercise_state"],
            ("topic", "difficulty_delta", "difficulty", "exercise_type", "template_id"),
        )
    else:
        result["exercise_state"] = None
    if result.get("clarification") is not None:
        result["clarification"] = _project_model(
            PublicClarification, result["clarification"], ("missing",)
        )
    else:
        result["clarification"] = None
    result.setdefault("cached", False)
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        public_metrics = {
            key: metrics[key] for key in ("tool_calls", "latency_ms") if key in metrics
        }
        result["metrics"] = PublicMetrics.model_validate(public_metrics).model_dump(
            mode="json", exclude_unset=True
        )
    else:
        result["metrics"] = {}
    result["response_type"] = response_type
    if response_type in {"verified_answer", "guided_exercise"}:
        _store_skill_contract(evidence, private_exercise)
    return ResponseEnvelope.model_validate(result).model_dump(mode="json", exclude_unset=True)

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

