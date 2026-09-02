# -*- coding: utf-8 -*-
"""FastAPI service with caching, feedback collection, and metrics."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import json
import logging
import secrets
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from agentic_rag import memory
from agentic_rag.cache import answer_cache
from agentic_rag.exercises.models import PublicExerciseState as AdaptiveExerciseState
from agentic_rag.domain.schemas import AttachmentParseOutput
from agentic_rag.graph import build_named_skill_pipeline_graph, build_skill_pipeline_graph
from agentic_rag.fast_path import build_fast_response
from agentic_rag.local_intents import parse_local_command
from agentic_rag.knowledge_graph import math_knowledge_graph
from agentic_rag.metrics import FEEDBACK, observe_state
from agentic_rag.response_contract import (
    capture_skill_contract, clarification_response, consume_skill_contract, normalize_response,
    peek_skill_contract, private_exercise_payload, public_response_digest,
    response_validation_digest,
    restore_validated_exercise_state,
)
from agentic_rag.reliability import FailureKind, resolve_failure
from agentic_rag.tracing import append_bad_case, persist_trace
from agentic_rag.skill_runtime.contracts import SkillContext, SkillStatus
from agentic_rag.skill_runtime.executor import SkillExecutor
from agentic_rag.skill_runtime.registry import get_default_registry
from config import (
    CHROMA_PATH,
    KNOWLEDGE_SOURCE_PATH,
    OPENAI_API_KEY,
    OPERATIONS_METRICS_TOKEN,
    RUN_TIMEOUT_SECONDS,
    FORCE_LLM_EVERY_TURN,
    LLM_MODEL_NAME,
    MODEL_PROVIDER,
    ENABLE_DENSE_RETRIEVAL,
    MATH_EMBEDDING_MODEL_PATH,
    TAVILY_API_KEY,
)

STATIC_DIR = Path(__file__).with_name("static")
_graph = None
_graph_lock = Lock()
_attachment_graph = None
_attachment_graph_lock = Lock()
_graph_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="math-agent")
_feedback_lock = Lock()
_skill_registry = get_default_registry()
_skill_executor = SkillExecutor(_skill_registry)
logger = logging.getLogger(__name__)
_PUBLIC_RESPONSE_TYPES = {
    "verified_answer",
    "guided_exercise",
    "clarification_required",
    "supported_refusal",
    "general_answer",
}
_GRAPH_VALIDATION_MODES = {
    "llm": "independent_critic",
    "local_sympy": "deterministic",
}
_ATTACHMENT_MEDIA_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "application/pdf",
}
_MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024
_ATTACHMENT_TIMEOUT_SECONDS = 70


@asynccontextmanager
async def lifespan(_app: FastAPI):
    memory.initialize_memory_db()
    yield
    _graph_executor.shutdown(wait=False, cancel_futures=True)


def _get_graph_executor() -> ThreadPoolExecutor:
    global _graph_executor
    if getattr(_graph_executor, "_shutdown", False):
        _graph_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="math-agent")
    return _graph_executor


app = FastAPI(title="初中数学错题智能问答系统", version="2025.2", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    language: Literal["zh", "en"] = "zh"
    conversation_history: list[dict[str, str]] = Field(default_factory=list, max_length=24)
    conversation_summary: str = ""
    exercise_state: AdaptiveExerciseState | None = None

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        return unicodedata.normalize("NFKC", value).strip()

    @field_validator("conversation_history")
    @classmethod
    def validate_history(cls, value: list[dict[str, str]]) -> list[dict[str, str]]:
        total = 0
        cleaned = []
        for item in value:
            if item.get("role") not in {"student", "tutor"}:
                raise ValueError("conversation_history role 必须是 student 或 tutor")
            content = unicodedata.normalize("NFKC", str(item.get("content", ""))).strip()
            if not content or len(content) > 8000:
                raise ValueError("conversation_history 单条内容长度必须为 1-8000")
            total += len(content)
            cleaned.append({"role": item["role"], "content": content})
        if total > 32000:
            raise ValueError("conversation_history 总长度不能超过 32000")
        return cleaned


class FeedbackRequest(BaseModel):
    trace_id: str
    correct: bool
    comment: str = Field(default="", max_length=2000)


@dataclass(frozen=True)
class CurriculumSkillRun:
    response: dict[str, Any]
    contract: dict[str, Any]

def get_graph():
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                _graph = build_skill_pipeline_graph(_skill_executor)
    return _graph


def get_attachment_graph():
    global _attachment_graph
    if _attachment_graph is None:
        with _attachment_graph_lock:
            if _attachment_graph is None:
                _attachment_graph = build_named_skill_pipeline_graph(
                    "multimodal",
                    _skill_executor,
                    feature_flags={"attachment_agent": True},
                )
    return _attachment_graph


def _run_attachment_skill(
    payload: dict[str, Any], trace_id: str
) -> tuple[AttachmentParseOutput | None, str]:
    graph_state = get_attachment_graph().invoke(
        {
            **payload,
            "trace_id": trace_id,
            "response_language": payload.get("language", "zh"),
            "deadline_at": (
                datetime.now(timezone.utc)
                + timedelta(seconds=_ATTACHMENT_TIMEOUT_SECONDS)
            ).timestamp(),
        },
        {"recursion_limit": 8},
    )
    pipeline_state = graph_state.get("skill_pipeline", {})
    return (
        pipeline_state.get("attachment_structure"),
        str(pipeline_state.get("safe_error", "")),
    )


def _run_curriculum_skill(request: AskRequest) -> CurriculumSkillRun | None:
    """Compatibility name for the single production Skill Pipeline entry."""
    trace_id = str(uuid.uuid4())
    context = SkillContext(
        request_id=trace_id,
        trace_id=trace_id,
        language=request.language,
        deadline_at=datetime.now(timezone.utc) + timedelta(seconds=RUN_TIMEOUT_SECONDS),
    )
    with capture_skill_contract() as capture:
        graph_state = get_graph().invoke(
            {
                "query": request.query,
                "response_language": request.language,
                "language": request.language,
                "conversation_summary": request.conversation_summary,
                "conversation_history": request.conversation_history,
                "exercise_state": (
                    request.exercise_state.model_dump(mode="json")
                    if request.exercise_state is not None
                    else None
                ),
                "trace_id": trace_id,
                "deadline_at": context.deadline_at.timestamp(),
            },
            {"recursion_limit": 16},
        )
        contract = consume_skill_contract(capture)
    # The Pipeline re-renders the typed envelope (for example by adding its
    # internal language field). Keep the trusted proof, then bind a fresh
    # public digest after the API performs its final projection.
    contract.pop("public_response_sha256", None)
    state = graph_state.get("skill_pipeline", {})
    response = state.get("response_render")
    if response is None:
        return None
    critic = state.get("answer_repair_critic") or state.get("answer_critic")
    if not contract and critic is not None and getattr(critic, "passed", False):
        mode = getattr(critic, "critic", {}).get("validation_mode")
        kind = "independent_critic" if mode == "llm" else "deterministic"
        contract = {"validation_evidence": {"kind": kind, "passed": True}}
    if contract.get("validation_evidence") is not None:
        response_payload = response.model_dump(
            mode="json", exclude_unset=True, exclude_none=True
        )
        response_payload["validation_evidence"] = contract["validation_evidence"]
        projected = normalize_response(
            response_payload,
            response_payload["response_type"],
            private_exercise=restore_validated_exercise_state(
                contract.get("private_exercise_state")
            ),
        )
        contract["public_response_sha256"] = public_response_digest(projected)
    return CurriculumSkillRun(
        response=response.model_dump(
            mode="json", exclude_unset=True, exclude_none=True
        ),
        contract=contract,
    )


def _route_cached_turn(request: AskRequest) -> bool:
    """Keep response-cache hits inside the observable multi-turn routing contract."""
    trace_id = str(uuid.uuid4())
    context = SkillContext(
        request_id=trace_id,
        trace_id=trace_id,
        language=request.language,
        deadline_at=datetime.now(timezone.utc) + timedelta(seconds=1),
    )
    result = _skill_executor.execute(
        "math.turn_router@1",
        {
            "query": request.query,
            "language": request.language,
            "conversation_summary": request.conversation_summary,
            "conversation_history": request.conversation_history,
            "exercise_state": (
                request.exercise_state.model_dump(mode="json")
                if request.exercise_state is not None
                else None
            ),
        },
        context,
        pipeline="math.correction.cache_hit@1.0.0",
    )
    return result.status == SkillStatus.OK


def _peek_skill_contract() -> dict[str, Any] | None:
    return peek_skill_contract()

def _document_sources(state: dict) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": doc.metadata.get("chunk_id"),
            "source": doc.metadata.get("source"),
            "chapter": doc.metadata.get("chapter"),
            "rank": doc.metadata.get("rank"),
            "excerpt": str(getattr(doc, "page_content", ""))[:1200] or None,
        }
        for doc in state.get("documents", [])
    ]


def _record_failure(trace_id: str, query: str, category: str, issues: list[str], started: float) -> None:
    state = {
        "trace_id": trace_id,
        "query": query,
        "response": "",
        "validation_passed": False,
        "validation_issues": issues,
        "bad_case_category": category,
        "metrics": {"tool_calls": 0, "critic_failures": 1, "hallucinations_detected": 0},
        "trace_events": [
            {"node": "start", "at": started, "query": query},
            {"node": category, "kind": "failed", "at": time.time(), "latency_ms": round((time.time() - started) * 1000, 2), "payload": {"issues": issues}},
        ],
    }
    for diagnostic in (
        lambda: persist_trace(state),
        lambda: observe_state(state, time.time() - started),
    ):
        try:
            diagnostic()
        except Exception:
            logger.exception("failure diagnostics could not be recorded")


class ContractViolation(ValueError):
    pass


class _CacheValidationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    kind: Literal["deterministic", "independent_critic"]
    passed: bool

    @model_validator(mode="after")
    def validate_passed(self) -> "_CacheValidationEvidence":
        if self.passed is not True:
            raise ValueError("cached validation evidence must have passed")
        return self


class _CachePrivateExerciseState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    template_id: str
    topic: str
    public_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    hidden_answer: str
    validation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_digest(self) -> "_CachePrivateExerciseState":
        if restore_validated_exercise_state(self.model_dump()) is None:
            raise ValueError("private exercise state has an invalid digest")
        return self


class _CacheContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    validation_evidence: _CacheValidationEvidence
    private_exercise_state: _CachePrivateExerciseState | None = None
    public_response_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )


class _CacheConversationTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    role: Literal["student", "tutor"]
    content: str


class _CacheSource(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    chunk_id: str | None = None
    source: str = ""
    chapter: str | None = None
    rank: int | float | None = None
    excerpt: str | None = Field(default=None, max_length=1200)


class _CacheExerciseState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    exercise_id: str | None = None
    session_id: str | None = None
    topic: str = ""
    grade: int | None = Field(default=None, ge=7, le=9)
    difficulty_delta: int = 0
    difficulty: int | None = None
    exercise_type: str | None = None
    template_id: str | None = None
    fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    knowledge_points: list[str] = Field(default_factory=list)


class _CacheClarification(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    missing: list[str] = Field(default_factory=list)


class _CacheMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    tool_calls: int | float | None = None
    model_attempts: int | float | None = None
    model_successes: int | float | None = None
    model_failures: int | float | None = None
    latency_ms: int | float | None = None
    model_provider: str | None = None
    model_name: str | None = None
    llm_required: bool | None = None
    execution_path: str | None = None

    @model_validator(mode="after")
    def validate_present_metrics_are_numeric(self) -> "_CacheMetrics":
        numeric_fields = {
            "tool_calls",
            "model_attempts",
            "model_successes",
            "model_failures",
            "latency_ms",
        }
        if any(
            getattr(self, field) is None
            for field in self.model_fields_set & numeric_fields
        ):
            raise ValueError("present cache metrics must be numeric")
        return self


class _CachePublicResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    answer: str
    trace_id: str
    intent: str
    knowledge_points: list[str]
    sources: list[_CacheSource]
    validation_passed: bool
    conversation_history: list[_CacheConversationTurn]
    conversation_summary: str
    exercise_state: _CacheExerciseState | None
    clarification: _CacheClarification | None
    cached: bool
    metrics: _CacheMetrics
    response_type: Literal["verified_answer", "guided_exercise"]

    @model_validator(mode="after")
    def validate_fixed_public_values(self) -> "_CachePublicResponse":
        if self.validation_passed is not True or self.cached is not False:
            raise ValueError("cached public response has invalid fixed values")
        return self


class _CacheRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    public: _CachePublicResponse
    contract: _CacheContract

    @model_validator(mode="after")
    def validate_response_type_matches_hidden_state(self) -> "_CacheRecord":
        expects_private_exercise = self.public.response_type == "guided_exercise"
        private_exercise = self.contract.private_exercise_state
        has_private_exercise = private_exercise is not None
        if has_private_exercise is not expects_private_exercise:
            raise ValueError("cached response type and hidden-answer state disagree")
        public_payload = self.public.model_dump(mode="json", exclude_unset=True)
        expected_digest = public_response_digest(public_payload)
        if self.contract.public_response_sha256 is None or not secrets.compare_digest(
            self.contract.public_response_sha256, expected_digest
        ):
            raise ValueError("cached public response digest does not match")
        if expects_private_exercise:
            public_exercise = self.public.exercise_state
            if (
                public_exercise is None
                or not public_exercise.template_id
                or not public_exercise.topic
                or not public_exercise.fingerprint
                or public_exercise.template_id != private_exercise.template_id
                or public_exercise.topic != private_exercise.topic
                or public_exercise.fingerprint != private_exercise.public_fingerprint
            ):
                raise ValueError("cached public and private exercise state disagree")
        return self


def _validated_cache_record(candidate: Any) -> dict[str, Any]:
    """Apply the one cache schema used by both the writer and reader."""
    try:
        return _CacheRecord.model_validate(candidate).model_dump(exclude_unset=True)
    except ValidationError as exc:
        raise ContractViolation("cache record must match the writer schema") from exc


def _validated_contract(contract: dict[str, Any] | None) -> dict[str, Any]:
    try:
        validated = _CacheContract.model_validate(contract)
    except (ValidationError, TypeError):
        return {}
    return validated.model_dump(mode="json", exclude_none=True)


def _graph_contract(state: dict[str, Any]) -> dict[str, Any]:
    critic = state.get("critic_report")
    response = state.get("response")
    mode = critic.get("validation_mode") if isinstance(critic, dict) else None
    deterministic = critic.get("deterministic") if isinstance(critic, dict) else None
    if (
        not isinstance(critic, dict)
        or not isinstance(response, str)
        or not response
        or state.get("response_type") != "verified_answer"
        or state.get("needs_clarification") is True
        or state.get("validation_passed") is not True
        or critic.get("is_valid") is not True
        or mode not in _GRAPH_VALIDATION_MODES
        or not isinstance(deterministic, dict)
        or deterministic.get("passed") is not True
        or state.get("draft_response") != response
        or critic.get("validated_response_sha256") != response_validation_digest(response)
    ):
        return {}
    kind = _GRAPH_VALIDATION_MODES[mode]
    return _validated_contract({"validation_evidence": {"kind": kind, "passed": True}})


def _clarification_for(request: AskRequest, trace_id: str | None = None) -> dict[str, Any]:
    missing = (
        ["the full problem, diagram conditions, or the step you want checked"]
        if request.language == "en"
        else ["完整题干、图形条件或需要核对的步骤"]
    )
    response = clarification_response(
        request.query,
        request.conversation_history,
        request.conversation_summary,
        missing,
        request.language,
    )
    if trace_id:
        response["trace_id"] = trace_id
    return response


def _reliability_response(
    request: AskRequest,
    failure_kind: FailureKind,
    trace_id: str,
) -> dict[str, Any]:
    payload = resolve_failure(
        query=request.query,
        language=request.language,
        history=request.conversation_history,
        summary=request.conversation_summary,
        failure_kind=failure_kind,
        issues=[],
    )
    payload["trace_id"] = trace_id
    return _public_response(payload, request)


def _verified_failure_fallback(
    request: AskRequest,
    trace_id: str,
) -> dict[str, Any] | None:
    """Keep a verified local result when upstream Agents exhaust their budget."""
    baseline = build_fast_response(
        request.query,
        request.conversation_history,
        request.conversation_summary,
        request.language,
        request.exercise_state,
    )
    if (
        not baseline
        or baseline.get("response_type") != "verified_answer"
        or baseline.get("validation_passed") is not True
    ):
        return None
    payload = dict(baseline)
    payload["trace_id"] = trace_id
    metrics = dict(payload.get("metrics") or {})
    successes = int(metrics.get("model_successes") or metrics.get("tool_calls") or 0)
    attempts = max(int(metrics.get("model_attempts") or successes), successes + 1)
    payload["metrics"] = {
        **metrics,
        "tool_calls": successes,
        "model_attempts": attempts,
        "model_successes": successes,
        "model_failures": max(
            int(metrics.get("model_failures") or 0),
            attempts - successes,
        ),
    }
    return _public_response(
        payload,
        request,
        contract={
            "validation_evidence": {
                "kind": "deterministic",
                "passed": True,
            }
        },
    )


def _require_public_digest_match(
    response: dict[str, Any], trusted_contract: dict[str, Any]
) -> dict[str, Any]:
    trusted_digest = trusted_contract.get("public_response_sha256")
    if trusted_digest is not None and not secrets.compare_digest(
        trusted_digest, public_response_digest(response)
    ):
        raise ContractViolation("response contract does not match public response")
    return response


def _bind_contract_to_public_response(
    contract: dict[str, Any], public_response: dict[str, Any]
) -> dict[str, Any]:
    """Bind trusted evidence after the API adds server-owned public metrics."""
    trusted_contract = _validated_contract(contract)
    if trusted_contract.get("validation_evidence") is None:
        raise ContractViolation("response contract is missing validation evidence")
    rebound = _validated_contract(
        {
            **trusted_contract,
            "public_response_sha256": public_response_digest(public_response),
        }
    )
    if not rebound:
        raise ContractViolation("response contract could not bind public response")
    return rebound


def _public_response(
    payload: dict[str, Any], request: AskRequest, *, contract: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Re-run the response contract before publishing any producer or cache output."""
    response_type = payload.get("response_type")
    if response_type is not None and response_type not in _PUBLIC_RESPONSE_TYPES:
        raise ContractViolation("unsupported response type")
    normalized_payload = dict(payload)
    for field in (
        "validation_evidence",
        "_validation_evidence",
        "exercise_answer_hidden",
        "_exercise_answer_hidden",
        "critic_report",
    ):
        normalized_payload.pop(field, None)
    trusted_contract = _validated_contract(contract)
    evidence = trusted_contract.get("validation_evidence")
    private_exercise = restore_validated_exercise_state(
        trusted_contract.get("private_exercise_state")
    )

    if response_type in {"verified_answer", "guided_exercise"}:
        if evidence is None or (
            response_type == "guided_exercise" and private_exercise is None
        ):
            raise ContractViolation("verified response is missing required evidence")
        normalized_payload["validation_evidence"] = evidence
        try:
            response = _require_public_digest_match(
                normalize_response(
                    normalized_payload,
                    response_type,
                    private_exercise=private_exercise,
                ),
                trusted_contract,
            )
            return _attach_runtime_metadata(response)
        except ValueError as exc:
            raise ContractViolation("response contract rejected producer payload") from exc
    if response_type in {"clarification_required", "supported_refusal", "general_answer"}:
        try:
            return _attach_runtime_metadata(
                normalize_response(normalized_payload, response_type)
            )
        except ValueError as exc:
            raise ContractViolation("response contract rejected producer payload") from exc
    if private_exercise is not None:
        if evidence is None:
            raise ContractViolation("guided response is missing validation evidence")
        normalized_payload["validation_evidence"] = evidence
        return _attach_runtime_metadata(_require_public_digest_match(
            normalize_response(
                normalized_payload,
                "guided_exercise",
                private_exercise=private_exercise,
            ),
            trusted_contract,
        ))
    if evidence is not None:
        normalized_payload["validation_evidence"] = evidence
        return _attach_runtime_metadata(
            _require_public_digest_match(
                normalize_response(normalized_payload, "verified_answer"),
                trusted_contract,
            )
        )
    if normalized_payload.get("clarification") or normalized_payload.get("needs_clarification"):
        return _attach_runtime_metadata(
            _clarification_for(request, normalized_payload.get("trace_id"))
        )
    if normalized_payload.get("refusal_reason"):
        normalized_payload["answer"] = normalized_payload["refusal_reason"]
        return _attach_runtime_metadata(
            normalize_response(normalized_payload, "supported_refusal")
        )
    return _attach_runtime_metadata(
        _clarification_for(request, normalized_payload.get("trace_id"))
    )


def _attach_runtime_metadata(response: dict[str, Any]) -> dict[str, Any]:
    """Add server-owned execution facts after producer evidence is verified."""
    published = dict(response)
    metrics = dict(published.get("metrics") or {})
    attempts = int(metrics.get("model_attempts") or 0)
    successes = int(metrics.get("model_successes") or 0)
    if published.get("response_type") == "guided_exercise":
        execution_path = "exercise_agent"
    elif published.get("intent") == "utility_time":
        execution_path = "utility_tool"
    elif published.get("response_type") == "general_answer":
        execution_path = "general_agent"
    elif published.get("sources"):
        execution_path = "agentic_rag"
    elif published.get("intent") in {"problem_solve", "multiple_choice"} and successes > 0:
        execution_path = "solve_agent"
    elif published.get("intent") == "error_analysis" and successes > 0:
        execution_path = "audit_agent"
    elif successes > 0:
        execution_path = "tutor_agent"
    elif attempts > 0:
        execution_path = "verified_fallback"
    elif published.get("intent") == "security_refusal":
        execution_path = "policy_guard"
    else:
        execution_path = "turn_router"
    published["metrics"] = {
        **metrics,
        "model_provider": MODEL_PROVIDER,
        "model_name": LLM_MODEL_NAME,
        "llm_required": FORCE_LLM_EVERY_TURN,
        "execution_path": execution_path,
    }
    return published


def _cache_record(public_response: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    trusted_contract = _validated_contract(contract)
    evidence = trusted_contract.get("validation_evidence")
    response_type = public_response.get("response_type")
    if response_type not in {"verified_answer", "guided_exercise"} or evidence is None:
        raise ContractViolation("cache record is missing validation evidence")
    private_exercise = restore_validated_exercise_state(
        trusted_contract.get("private_exercise_state")
    )
    if response_type == "guided_exercise" and private_exercise is None:
        raise ContractViolation("cache record is missing hidden-answer signal")
    if response_type == "verified_answer" and private_exercise is not None:
        raise ContractViolation("verified cache record cannot carry exercise state")
    trusted_digest = trusted_contract.get("public_response_sha256")
    expected_digest = public_response_digest(public_response)
    if trusted_digest is None or not secrets.compare_digest(
        trusted_digest, expected_digest
    ):
        raise ContractViolation("cache record does not match trusted public response")
    contract_payload: dict[str, Any] = {
        "validation_evidence": evidence,
        "public_response_sha256": trusted_digest,
    }
    if private_exercise is not None:
        contract_payload["private_exercise_state"] = private_exercise_payload(
            private_exercise
        )
    return _validated_cache_record(
        {
            "public": public_response,
            "contract": contract_payload,
        }
    )


def _cached_response(cached: Any, request: AskRequest) -> dict[str, Any]:
    record = _validated_cache_record(cached)
    return _public_response(record["public"], request, contract=record["contract"])


def _bypass_answer_cache(request: AskRequest) -> bool:
    if FORCE_LLM_EVERY_TURN:
        return True
    if request.exercise_state is not None:
        return True
    command = parse_local_command(request.query, request.language)
    if command is not None and command.action in {
        "practice",
        "next_exercise",
        "adjust_difficulty",
    }:
        return True
    normalized = unicodedata.normalize("NFKC", request.query).lower()
    has_topic = any(
        marker in normalized
        for marker in ("几何", "代数", "一次函数", "geometry", "algebra", "linear function")
    )
    asks_for_practice = any(
        marker in normalized
        for marker in ("练习", "来一道", "出一道", "生成", "give me", "practice")
    )
    return has_topic and asks_for_practice


def _cacheable_response(response: dict[str, Any], cache_enabled: bool) -> bool:
    if not cache_enabled or response.get("response_type") not in {
        "verified_answer",
        "guided_exercise",
    }:
        return False
    exercise_state = response.get("exercise_state")
    return not (
        isinstance(exercise_state, dict) and exercise_state.get("exercise_id")
    )


def _readiness_checks() -> dict[str, bool]:
    vector_database = Path(CHROMA_PATH) / "chroma.sqlite3"
    vector_ready = False
    if vector_database.is_file():
        try:
            import sqlite3

            with sqlite3.connect(vector_database) as connection:
                row = connection.execute(
                    """SELECT COUNT(e.id)
                       FROM embeddings e
                       JOIN segments s ON s.id = e.segment_id
                       JOIN collections c ON c.id = s.collection
                       WHERE c.name = ?""",
                    ("math_chunks",),
                ).fetchone()
            vector_ready = bool(row and row[0] > 0)
        except (OSError, sqlite3.DatabaseError):
            vector_ready = False
    return {
        "static_ui": (STATIC_DIR / "index.html").exists(),
        "knowledge_source": Path(KNOWLEDGE_SOURCE_PATH).is_file(),
        "vector_index": vector_ready,
        "model_configured": bool(OPENAI_API_KEY),
        "local_curriculum": True,
    }


def _math_chunk_count() -> int:
    vector_database = Path(CHROMA_PATH) / "chroma.sqlite3"
    if not vector_database.is_file():
        return 0
    try:
        import sqlite3

        with sqlite3.connect(vector_database) as connection:
            row = connection.execute(
                """SELECT COUNT(e.id)
                   FROM embeddings e
                   JOIN segments s ON s.id = e.segment_id
                   JOIN collections c ON c.id = s.collection
                   WHERE c.name = ?""",
                ("math_chunks",),
            ).fetchone()
        return int(row[0]) if row else 0
    except (OSError, sqlite3.DatabaseError, TypeError, ValueError):
        return 0


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    return {"status": "ready" if all(_readiness_checks().values()) else "degraded"}


@app.get("/runtime")
def runtime():
    return {
        "model": {
            "configured": bool(OPENAI_API_KEY),
            "provider": MODEL_PROVIDER,
            "name": LLM_MODEL_NAME,
            "force_every_math_turn": FORCE_LLM_EVERY_TURN,
        },
        "retrieval": {
            "chunk_count": _math_chunk_count(),
            "dense_enabled": ENABLE_DENSE_RETRIEVAL,
            "embedding_model": MATH_EMBEDDING_MODEL_PATH,
            "graph_nodes": len(math_knowledge_graph.as_dict()["prerequisites"]),
            "web_search_configured": bool(TAVILY_API_KEY),
        },
    }

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.post("/attachments/parse")
async def parse_attachment(
    file: UploadFile = File(...),
    language: Literal["zh", "en"] = Form("zh"),
):
    trace_id = str(uuid.uuid4())
    media_type = str(file.content_type or "").lower()
    if media_type not in _ATTACHMENT_MEDIA_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                "Only JPG, PNG, WebP, and PDF files are supported."
                if language == "en"
                else "仅支持 JPG、PNG、WebP 和 PDF 文件。"
            ),
        )

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > _MAX_ATTACHMENT_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    "The attachment must be smaller than 8 MB."
                    if language == "en"
                    else "附件必须小于 8 MB。"
                ),
            )
        chunks.append(chunk)
    content = b"".join(chunks)
    if not content:
        raise HTTPException(
            status_code=400,
            detail="The attachment is empty." if language == "en" else "附件内容为空。",
        )
    filename = Path(file.filename or "attachment").name
    payload = {
        "filename": filename,
        "media_type": media_type,
        "content_base64": base64.b64encode(content).decode("ascii"),
        "language": language,
    }
    loop = asyncio.get_running_loop()
    try:
        parsed, safe_error = await asyncio.wait_for(
            loop.run_in_executor(
                _get_graph_executor(), _run_attachment_skill, payload, trace_id
            ),
            timeout=_ATTACHMENT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                "Recognition took too long. The file was not saved; try a clearer image."
                if language == "en"
                else "识别时间过长，文件未保存；请裁剪后上传更清晰的图片。"
            ),
        ) from exc
    if parsed is None:
        raise HTTPException(
            status_code=400,
            detail=safe_error
            or (
                "The attachment could not be read."
                if language == "en"
                else "附件无法读取，请检查文件后重试。"
            ),
        )
    return {
        **parsed.model_dump(mode="json"),
        "trace_id": trace_id,
    }


@app.post("/ask")
async def ask(request: AskRequest):
    started = time.time()
    try:
        cache_payload = request.model_dump()
        cache_enabled = not _bypass_answer_cache(request)
        cached = answer_cache.get(cache_payload) if cache_enabled else None
        if cached is not None and _route_cached_turn(request):
            response = _cached_response(cached, request)
            observe_state({"response": response["answer"], "metrics": response["metrics"]}, 0)
            return {**response, "cached": True}

        loop = asyncio.get_running_loop()
        fast_run = await asyncio.wait_for(
            loop.run_in_executor(_get_graph_executor(), _run_curriculum_skill, request),
            timeout=RUN_TIMEOUT_SECONDS,
        )
        if fast_run:
            fast_payload = fast_run.response
            fast_contract = _validated_contract(fast_run.contract)
            response = _public_response(fast_payload, request, contract=fast_contract)
            latency = time.time() - started
            response["metrics"]["latency_ms"] = round(latency * 1000, 2)
            trace_state = {
                **response,
                "query": request.query,
                "response": response["answer"],
                "trace_events": [
                    {"node": "start", "at": started, "query": request.query},
                    {"node": "skill_pipeline", "kind": "completed", "at": time.time(), "latency_ms": response["metrics"]["latency_ms"], "payload": {"response_type": response["response_type"], "intent": response.get("intent")}},
                ],
                "validation_issues": [],
            }
            persist_trace(trace_state)
            observe_state(trace_state, latency)
            if _cacheable_response(response, cache_enabled):
                cache_contract = _bind_contract_to_public_response(
                    fast_contract, response
                )
                answer_cache.set(
                    cache_payload, _cache_record(response, cache_contract)
                )
            return response
        raise RuntimeError("production pipeline returned no response")
    except HTTPException:
        raise
    except asyncio.TimeoutError:
        trace_id = str(uuid.uuid4())
        _record_failure(trace_id, request.query, "timeout", ["timeout"], started)
        response = _verified_failure_fallback(request, trace_id) or _reliability_response(
            request, "timeout", trace_id
        )
        response["metrics"]["latency_ms"] = round((time.time() - started) * 1000, 2)
        return response
    except ContractViolation as exc:
        trace_id = str(uuid.uuid4())
        _record_failure(trace_id, request.query, "contract_error", [type(exc).__name__], started)
        response = _reliability_response(request, "runtime_error", trace_id)
        response["metrics"]["latency_ms"] = round((time.time() - started) * 1000, 2)
        return response
    except Exception as exc:
        trace_id = str(uuid.uuid4())
        _record_failure(trace_id, request.query, "runtime_error", [type(exc).__name__], started)
        response = _verified_failure_fallback(request, trace_id) or _reliability_response(
            request, "runtime_error", trace_id
        )
        response["metrics"]["latency_ms"] = round((time.time() - started) * 1000, 2)
        return response

@app.post("/feedback")
def feedback(request: FeedbackRequest):
    FEEDBACK.labels(str(request.correct).lower()).inc()
    path = Path("evaluation/pending_labels.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with _feedback_lock:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(request.model_dump(), ensure_ascii=False) + "\n")
    if not request.correct:
        append_bad_case(request.trace_id, "", "negative_feedback", [request.comment or "用户标记回答有问题"])
    return {"accepted": True}


@app.get("/metrics", include_in_schema=False)
def metrics(authorization: str | None = Header(default=None)):
    token = OPERATIONS_METRICS_TOKEN
    supplied = authorization or ""
    try:
        wire_bytes = supplied.encode("latin-1")
        supplied_bytes = wire_bytes.decode("utf-8").encode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        supplied_bytes = supplied.encode("utf-8", errors="surrogatepass")
    expected_bytes = (
        b"Bearer " + token.encode("utf-8", errors="surrogatepass") if token else b""
    )
    if not token or not secrets.compare_digest(supplied_bytes, expected_bytes):
        raise HTTPException(status_code=404, detail="Not Found")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
