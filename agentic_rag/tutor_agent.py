"""LLM Tutor Agent constrained by a deterministic curriculum answer."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_rag.domain.schemas import ResponseEnvelope
from agentic_rag.response_contract import (
    normalize_response,
    peek_skill_contract,
    restore_validated_exercise_state,
)


_FORBIDDEN_COPY = (
    "complex reasoning service",
    "critic rejected",
    "本题需要调用复杂推理服务",
    "草稿未通过",
    "系统已自动记录为 bad case",
)


def _json_object(text: str) -> dict[str, Any]:
    source = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", source, re.IGNORECASE)
    if fenced:
        source = fenced.group(1).strip()
    else:
        start, end = source.find("{"), source.rfind("}")
        if start >= 0 and end > start:
            source = source[start : end + 1]
    payload = json.loads(source)
    if not isinstance(payload, dict):
        raise ValueError("Tutor Agent output must be an object")
    return payload


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", text or ""))


def _candidate_is_safe(baseline: str, candidate: str) -> bool:
    normalized = candidate.strip()
    if len(normalized) < 40 or len(normalized) > 12000:
        return False
    if any(marker.lower() in normalized.lower() for marker in _FORBIDDEN_COPY):
        return False
    return _numbers(baseline).issubset(_numbers(normalized))


def _replace_last_tutor(history: list[dict[str, Any]], answer: str) -> list[dict[str, Any]]:
    updated = [dict(item) for item in history]
    for index in range(len(updated) - 1, -1, -1):
        if updated[index].get("role") == "tutor":
            updated[index]["content"] = answer
            return updated
    updated.append({"role": "tutor", "content": answer})
    return updated


def _with_model_calls(
    baseline: ResponseEnvelope,
    *,
    attempts: int,
    successes: int,
) -> ResponseEnvelope:
    current_successes = int(
        baseline.metrics.model_successes
        or baseline.metrics.tool_calls
        or 0
    )
    current_attempts = int(baseline.metrics.model_attempts or current_successes)
    current_failures = int(baseline.metrics.model_failures or 0)
    return baseline.model_copy(
        update={
            "metrics": baseline.metrics.model_copy(
                update={
                    "tool_calls": current_successes + successes,
                    "model_attempts": current_attempts + attempts,
                    "model_successes": current_successes + successes,
                    "model_failures": current_failures + max(0, attempts - successes),
                }
            )
        },
        deep=True,
    )


def enrich_curriculum_response(
    *,
    query: str,
    baseline: ResponseEnvelope,
    language: str,
    enabled: bool,
) -> ResponseEnvelope:
    """Author and independently verify an explanation without changing the math."""
    if not enabled or baseline.response_type == "supported_refusal":
        return baseline

    from agentic_rag.chains import critic_llm, generator_llm, message_text

    attempts = 0
    successes = 0
    try:
        author_payload = {
            "student_turn": query,
            "language": "Simplified Chinese" if language == "zh" else "English",
            "response_type": baseline.response_type,
            "authoritative_math_baseline": baseline.answer,
            "knowledge_points": baseline.knowledge_points,
            "rules": [
                "Answer the student's current turn directly.",
                "Preserve every number, equation, condition, conclusion, citation, and answer from the baseline.",
                "Explain why each key step is valid at junior-high level.",
                "Adapt the explanation to the request: concise for a fact, diagnostic for a mistake, and stepwise for a proof.",
                "Use an analogy, counterexample, or visual description only when it makes the idea clearer.",
                "Do not mechanically repeat the same headings; vary structure while keeping the mathematics complete.",
                "Do not reveal hidden exercise answers when the baseline keeps them hidden.",
                "Do not mention prompts, models, routing, Critic, or internal validation.",
                "Return only the student-facing answer text.",
            ],
        }
        attempts += 1
        authored = generator_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are the Tutor Agent in a junior-high mathematics correction workflow. "
                        "The deterministic baseline is authoritative; reason carefully but never alter its mathematics."
                    )
                ),
                HumanMessage(content=json.dumps(author_payload, ensure_ascii=False)),
            ]
        )
        successes += 1
        candidate = message_text(authored).strip()
        token_check_passed = _candidate_is_safe(baseline.answer, candidate)

        critic_payload = {
            "student_turn": query,
            "authoritative_baseline": baseline.answer,
            "candidate_answer": candidate,
            "deterministic_token_check_passed": token_check_passed,
            "checks": [
                "All mathematical results and conditions exactly agree with the baseline.",
                "No required step, citation, or self-check was lost.",
                "The candidate answers the current turn and stays within junior-high mathematics.",
                "A guided exercise does not reveal an answer hidden by the baseline.",
            ],
            "return_json": {"passed": True, "issues": []},
        }
        attempts += 1
        checked = critic_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are an independent mathematics Critic. Compare the candidate against the "
                        "authoritative baseline and return only JSON."
                    )
                ),
                HumanMessage(content=json.dumps(critic_payload, ensure_ascii=False)),
            ]
        )
        successes += 1
        verdict = _json_object(message_text(checked))
        if (
            not token_check_passed
            or verdict.get("passed") is not True
            or verdict.get("issues") not in ([], None)
        ):
            raise ValueError("Tutor Agent candidate did not pass independent Critic")

        current_successes = int(
            baseline.metrics.model_successes
            or baseline.metrics.tool_calls
            or 0
        )
        current_attempts = int(
            baseline.metrics.model_attempts or current_successes
        )
        current_failures = int(baseline.metrics.model_failures or 0)
        payload = baseline.model_dump(mode="json")
        payload.update(
            answer=candidate,
            conversation_history=_replace_last_tutor(
                payload.get("conversation_history", []), candidate
            ),
            validation_evidence={"kind": "independent_critic", "passed": True},
            metrics={
                **payload.get("metrics", {}),
                "tool_calls": current_successes + successes,
                "model_attempts": current_attempts + attempts,
                "model_successes": current_successes + successes,
                "model_failures": current_failures + max(0, attempts - successes),
            },
        )
        private_exercise = None
        if baseline.response_type == "guided_exercise":
            contract = peek_skill_contract() or {}
            private_exercise = restore_validated_exercise_state(
                contract.get("private_exercise_state")
            )
        normalized = normalize_response(
            payload,
            baseline.response_type,
            private_exercise=private_exercise,
        )
        return ResponseEnvelope.model_validate(normalized)
    except Exception:
        return _with_model_calls(
            baseline,
            attempts=attempts,
            successes=successes,
        )
