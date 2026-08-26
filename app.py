# -*- coding: utf-8 -*-
"""FastAPI service with caching, feedback collection, and metrics."""

from __future__ import annotations

import asyncio
import json
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field, field_validator

from agentic_rag import memory
from agentic_rag.cache import answer_cache
from agentic_rag.graph import build_graph
from agentic_rag.guardrails import input_guardrail_violation
from agentic_rag.metrics import FEEDBACK, observe_state
from agentic_rag.response_contract import clarification_response, normalize_response
from agentic_rag.tracing import append_bad_case, persist_trace
from agentic_rag.skill_runtime.contracts import SkillContext, SkillStatus
from agentic_rag.skill_runtime.executor import SkillExecutor
from agentic_rag.skill_runtime.registry import get_default_registry
from config import CHROMA_PATH, OPENAI_API_KEY, RUN_TIMEOUT_SECONDS

STATIC_DIR = Path(__file__).with_name("static")
_graph = None
_graph_lock = Lock()
_graph_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="math-agent")
_feedback_lock = Lock()
_skill_registry = get_default_registry()
_skill_executor = SkillExecutor(_skill_registry)
_PUBLIC_RESPONSE_TYPES = {
    "verified_answer",
    "guided_exercise",
    "clarification_required",
    "supported_refusal",
}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    memory.initialize_memory_db()
    yield
    _graph_executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="初中数学错题智能问答系统", version="2025.2", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    language: Literal["zh", "en"] = "zh"
    conversation_history: list[dict[str, str]] = Field(default_factory=list, max_length=24)
    conversation_summary: str = ""

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


def get_graph():
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                _graph = build_graph()
    return _graph


def _run_curriculum_skill(request: AskRequest) -> dict | None:
    trace_id = str(uuid.uuid4())
    context = SkillContext(
        request_id=trace_id,
        trace_id=trace_id,
        language=request.language,
        deadline_at=datetime.now(timezone.utc) + timedelta(seconds=RUN_TIMEOUT_SECONDS),
    )
    result = _skill_executor.execute(
        "math.curriculum_solve@1",
        {
            "query": request.query,
            "conversation_summary": request.conversation_summary,
            "conversation_history": request.conversation_history,
            "language": request.language,
        },
        context,
        pipeline="math.correction@1.0.0",
    )
    if result.status != SkillStatus.OK or not result.value or not result.value.handled:
        return None
    response = result.value.response
    if response:
        response.setdefault("metrics", {})["skill_runtime_ms"] = result.metrics["latency_ms"]
        response["metrics"]["skill"] = result.provenance
    return response


def _document_sources(state: dict) -> list[dict[str, Any]]:
    return [
        {"chunk_id": doc.metadata.get("chunk_id"), "source": doc.metadata.get("source"), "chapter": doc.metadata.get("chapter"), "rank": doc.metadata.get("rank")}
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
    persist_trace(state)
    observe_state(state, time.time() - started)


def _validation_evidence(payload: dict[str, Any]) -> dict[str, Any] | None:
    evidence = payload.get("validation_evidence")
    if isinstance(evidence, dict):
        return evidence
    critic = payload.get("critic_report")
    if not isinstance(critic, dict) or payload.get("validation_passed") is not True:
        return None
    if critic.get("is_valid") is not True:
        return None
    return {
        "kind": "deterministic" if str(critic.get("validation_mode", "")).startswith("local_") else "independent_critic",
        "passed": True,
    }


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


def _public_response(payload: dict[str, Any], request: AskRequest) -> dict[str, Any]:
    """Project a producer payload onto the browser-safe response contract."""
    response_type = payload.get("response_type")
    normalized_payload = dict(payload)
    if response_type in _PUBLIC_RESPONSE_TYPES:
        if response_type in {"verified_answer", "guided_exercise"}:
            normalized_payload.setdefault("validation_evidence", {"kind": "deterministic", "passed": True})
        if response_type == "guided_exercise":
            normalized_payload.setdefault("exercise_answer_hidden", True)
        return normalize_response(normalized_payload, response_type)

    evidence = _validation_evidence(normalized_payload)
    if normalized_payload.get("exercise_answer_hidden") is True or (
        isinstance(normalized_payload.get("critic_report"), dict)
        and normalized_payload["critic_report"].get("exercise_answer_hidden") is True
    ):
        if evidence:
            normalized_payload["validation_evidence"] = evidence
            normalized_payload["exercise_answer_hidden"] = True
            return normalize_response(normalized_payload, "guided_exercise")
        return _clarification_for(request, normalized_payload.get("trace_id"))
    if evidence:
        normalized_payload["validation_evidence"] = evidence
        return normalize_response(normalized_payload, "verified_answer")
    if normalized_payload.get("clarification") or normalized_payload.get("needs_clarification"):
        return _clarification_for(request, normalized_payload.get("trace_id"))
    if normalized_payload.get("refusal_reason"):
        normalized_payload["answer"] = normalized_payload["refusal_reason"]
        return normalize_response(normalized_payload, "supported_refusal")
    return _clarification_for(request, normalized_payload.get("trace_id"))


@app.get("/health")
def health():
    return {"status": "ok", "version": app.version, "local_curriculum": True, "agent_timeout_seconds": RUN_TIMEOUT_SECONDS}


@app.get("/ready")
def ready():
    checks = {
        "static_ui": (STATIC_DIR / "index.html").exists(),
        "knowledge_source": Path("data/初中数学核心知识.md").exists(),
        "vector_index": Path(CHROMA_PATH).exists(),
        "model_configured": bool(OPENAI_API_KEY),
        "local_curriculum": True,
    }
    return {"status": "ready" if all(checks.values()) else "degraded", "checks": checks}


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.post("/ask")
async def ask(request: AskRequest):
    request_started = time.time()
    guard_issue = input_guardrail_violation(request.query)
    if guard_issue:
        trace_id = str(uuid.uuid4())
        _record_failure(trace_id, request.query, "input_guardrail", [guard_issue], request_started)
        raise HTTPException(status_code=400, detail="输入不符合数学教学系统的安全约束，请只提交题目、解题步骤或相关追问。")
    cache_payload = request.model_dump()
    cached = answer_cache.get(cache_payload)
    if cached:
        response = _public_response(cached, request)
        observe_state({"response": response["answer"], "metrics": response["metrics"]}, 0)
        return {**response, "cached": True}
    started = time.time()
    fast_response = _run_curriculum_skill(request)
    if fast_response:
        latency = time.time() - started
        fast_response = _public_response(fast_response, request)
        fast_response["metrics"]["latency_ms"] = round(latency * 1000, 2)
        trace_state = {
            **fast_response,
            "query": request.query,
            "response": fast_response["answer"],
            "trace_events": [
                {"node": "start", "at": started, "query": request.query},
                {"node": "deterministic_router", "kind": "completed", "at": time.time(), "latency_ms": fast_response["metrics"]["latency_ms"], "payload": {"response_type": fast_response["response_type"]}},
            ],
            "validation_issues": [],
        }
        persist_trace(trace_state)
        observe_state(trace_state, latency)
        answer_cache.set(cache_payload, fast_response)
        return fast_response
    try:
        loop = asyncio.get_running_loop()
        state = await asyncio.wait_for(
            loop.run_in_executor(
                _graph_executor,
                get_graph().invoke,
                {
                    "query": request.query,
                    "response_language": request.language,
                    "conversation_history": request.conversation_history,
                    "conversation_summary": request.conversation_summary,
                    "correction_attempts": 0,
                    "validation_issues": [],
                },
                {"recursion_limit": 64},
            ),
            timeout=RUN_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        trace_id = str(uuid.uuid4())
        issue = f"Agent 未在 {RUN_TIMEOUT_SECONDS:g} 秒产品 SLA 内完成"
        _record_failure(trace_id, request.query, "timeout", [issue], started)
        response = _clarification_for(request, trace_id)
        response["metrics"]["latency_ms"] = round((time.time() - started) * 1000, 2)
        return response
    except Exception as exc:
        trace_id = str(uuid.uuid4())
        _record_failure(trace_id, request.query, "runtime_error", [type(exc).__name__], started)
        response = _clarification_for(request, trace_id)
        response["metrics"]["latency_ms"] = round((time.time() - started) * 1000, 2)
        return response
    latency = time.time() - started
    observe_state(state, latency)
    raw_response = {
        "answer": state.get("response", ""),
        "trace_id": state.get("trace_id"),
        "intent": state.get("intent"),
        "knowledge_points": state.get("knowledge_points", []),
        "sources": _document_sources(state),
        "validation_passed": state.get("validation_passed", False),
        "critic_report": state.get("critic_report", {}),
        "conversation_history": state.get("conversation_history", []),
        "conversation_summary": state.get("conversation_summary", ""),
        "metrics": state.get("metrics", {}),
        "latency_ms": round(latency * 1000, 2),
        "cached": False,
    }
    if response["validation_passed"]:
        answer_cache.set(cache_payload, response)
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


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
