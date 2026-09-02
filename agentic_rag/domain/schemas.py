"""Pydantic input/output contracts for repository-owned business Skills."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from agentic_rag.exercises.models import PublicExerciseState as AdaptiveExerciseState


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class QueryInput(StrictModel):
    query: str = Field(min_length=1, max_length=12000)
    conversation_summary: str = Field(default="", max_length=12000)


class CompletenessResult(StrictModel):
    status: Literal[
        "complete",
        "missing_conditions",
        "requires_image",
        "out_of_scope",
    ]
    missing: list[str] = Field(default_factory=list)
    follow_up: str = ""


AttachmentMediaType = Literal[
    "image/jpeg",
    "image/png",
    "image/webp",
    "application/pdf",
]


class AttachmentUploadInput(StrictModel):
    filename: str = Field(min_length=1, max_length=200)
    media_type: AttachmentMediaType
    content_base64: str = Field(min_length=4, max_length=12_000_000)
    language: Literal["zh", "en"] = "zh"


class AttachmentExtractOutput(StrictModel):
    filename: str
    media_type: AttachmentMediaType
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(ge=1, le=8 * 1024 * 1024)
    extracted_text: str = Field(default="", max_length=24000)
    page_count: int = Field(default=1, ge=1, le=10)
    image_width: int | None = Field(default=None, ge=1)
    image_height: int | None = Field(default=None, ge=1)
    warnings: list[str] = Field(default_factory=list)


class AttachmentStructureInput(AttachmentUploadInput):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(ge=1, le=8 * 1024 * 1024)
    extracted_text: str = Field(default="", max_length=24000)
    page_count: int = Field(default=1, ge=1, le=10)
    image_width: int | None = Field(default=None, ge=1)
    image_height: int | None = Field(default=None, ge=1)
    warnings: list[str] = Field(default_factory=list)


class AttachmentParseOutput(StrictModel):
    status: Literal["ready", "needs_confirmation", "unsupported"]
    filename: str
    media_type: AttachmentMediaType
    problem_text: str = Field(default="", max_length=8000)
    student_answer: str = Field(default="", max_length=3000)
    formulas: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    page_count: int = Field(default=1, ge=1, le=10)
    parser: Literal["vision_agent", "pdf_text", "manual_confirmation"]


class GuardOutput(StrictModel):
    normalized_query: str
    allowed: bool = True
    reason: str = ""


class QuestionParseOutput(StrictModel):
    stem: str
    student_answer: str = ""
    intent: str = "solve"
    error_clues: list[str] = Field(default_factory=list)


class QueryRewriteInput(QueryInput):
    stem: str = ""
    intent: str = "solve"


class QueryRewriteOutput(StrictModel):
    rewritten_query: str
    sub_queries: list[str]
    missing_conditions: list[str] = Field(default_factory=list)


class ClassificationOutput(StrictModel):
    grade: str = ""
    chapter: str = "综合"
    knowledge_points: list[str] = Field(default_factory=list)
    question_type: str = ""


TurnIntent = Literal[
    "answer_submission",
    "error_analysis",
    "conceptual_followup",
    "hint_request",
    "solution_reveal",
    "new_exercise",
    "difficulty_adjustment",
    "knowledge_query",
    "multiple_choice",
    "problem_solve",
    "problem_switch",
    "out_of_scope",
    "utility_query",
    "general_chat",
]


class CurriculumSolveInput(QueryInput):
    language: Literal["zh", "en"] = "zh"
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    exercise_state: AdaptiveExerciseState | None = None
    intent: TurnIntent | None = None


class TurnRouteInput(CurriculumSolveInput):
    pass


class TurnRouteOutput(StrictModel):
    intent: TurnIntent
    route: Literal[
        "deterministic",
        "exercise_agent",
        "rag",
        "solve_agent",
        "audit_agent",
        "scope_response",
        "utility_tool",
        "general_agent",
    ]
    routed_query: str
    has_active_exercise: bool = False
    reason: str = ""


class RetrievalInput(StrictModel):
    query: str = Field(min_length=1, max_length=12000)
    sub_queries: list[str] = Field(default_factory=list)
    chapter: str = ""
    knowledge_points: list[str] = Field(default_factory=list)
    top_k: int = Field(default=6, ge=1, le=50)


class RetrievalCandidate(StrictModel):
    chunk_id: str | None = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    source: str = ""


class RetrievalOutput(StrictModel):
    candidates: list[RetrievalCandidate] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    model_attempts: int = Field(default=0, ge=0)
    model_successes: int = Field(default=0, ge=0)
    model_failures: int = Field(default=0, ge=0)


class FusionInput(StrictModel):
    rankings: list[list[RetrievalCandidate]] = Field(default_factory=list)
    top_k: int = Field(default=12, ge=1, le=100)


class RerankInput(StrictModel):
    query: str
    candidates: list[RetrievalCandidate]
    sub_queries: list[str] = Field(default_factory=list)
    knowledge_points: list[str] = Field(default_factory=list)
    top_k: int = Field(default=6, ge=1, le=50)


class AnswerGenerateInput(StrictModel):
    query: str
    student_answer: str = ""
    contexts: list[RetrievalCandidate] = Field(default_factory=list)
    intent: str = "knowledge_query"
    language: Literal["zh", "en"] = "zh"


class AnswerDraftOutput(StrictModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    model_attempts: int = Field(default=0, ge=0)
    model_successes: int = Field(default=0, ge=0)
    model_failures: int = Field(default=0, ge=0)


class AnswerRepairInput(StrictModel):
    query: str
    answer: str
    contexts: list[RetrievalCandidate] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    intent: str = ""
    student_answer: str = ""
    language: Literal["zh", "en"] = "zh"
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    conversation_summary: str = ""
    exercise_state: AdaptiveExerciseState | None = None


class AnswerCriticInput(StrictModel):
    query: str
    answer: str
    contexts: list[str] = Field(default_factory=list)
    student_answer: str = ""


class CriticOutput(StrictModel):
    passed: bool
    factual_faithfulness: bool
    math_logic_valid: bool
    issues: list[str] = Field(default_factory=list)
    deterministic: dict[str, Any] = Field(default_factory=dict)
    critic: dict[str, Any] = Field(default_factory=dict)
    model_attempts: int = Field(default=0, ge=0)
    model_successes: int = Field(default=0, ge=0)
    model_failures: int = Field(default=0, ge=0)


class SimilarExerciseInput(QueryInput):
    knowledge_points: list[str] = Field(default_factory=list)
    top_k: int = Field(default=3, ge=1, le=10)


class SimilarExerciseOutput(StrictModel):
    exercises: list[RetrievalCandidate] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)


class AnswerCheckInput(StrictModel):
    query: str
    student_answer: str
    contexts: list[str] = Field(default_factory=list)


class MemoryInput(StrictModel):
    session_id: str
    query: str = ""
    knowledge_points: list[str] = Field(default_factory=list)


class MemoryOutput(StrictModel):
    summary: str = ""
    events: list[dict[str, Any]] = Field(default_factory=list)


ResponseType = Literal[
    "verified_answer",
    "guided_exercise",
    "clarification_required",
    "supported_refusal",
    "general_answer",
]


class PublicSource(StrictModel):
    chunk_id: str | None = None
    source: str = ""
    chapter: str | None = None
    rank: int | float | None = None
    excerpt: str | None = Field(default=None, max_length=1200)


class PublicConversationTurn(StrictModel):
    role: Literal["student", "tutor"]
    content: str


class PublicExerciseState(StrictModel):
    exercise_id: str | None = None
    session_id: str | None = None
    topic: str = ""
    grade: int | None = Field(default=None, ge=7, le=9)
    difficulty_delta: int | None = None
    difficulty: int | None = None
    exercise_type: str | None = None
    template_id: str | None = None
    fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    knowledge_points: list[str] = Field(default_factory=list)


class PublicClarification(StrictModel):
    missing: list[str] = Field(default_factory=list)


class PublicMetrics(StrictModel):
    tool_calls: int | float | None = None
    model_attempts: int | float | None = None
    model_successes: int | float | None = None
    model_failures: int | float | None = None
    latency_ms: int | float | None = None
    model_provider: str | None = None
    model_name: str | None = None
    llm_required: bool | None = None
    execution_path: str | None = None


class ResponseEnvelope(StrictModel):
    response_type: ResponseType
    answer: str = ""
    trace_id: str = ""
    intent: str = ""
    knowledge_points: list[str] = Field(default_factory=list)
    sources: list[PublicSource] = Field(default_factory=list)
    validation_passed: bool = False
    conversation_history: list[PublicConversationTurn] = Field(default_factory=list)
    conversation_summary: str = ""
    exercise_state: PublicExerciseState | None = None
    clarification: PublicClarification | None = None
    metrics: PublicMetrics = Field(default_factory=PublicMetrics)
    cached: bool = False


class CurriculumSolveOutput(StrictModel):
    handled: bool
    response: ResponseEnvelope | None = None


class CurriculumTutorInput(CurriculumSolveInput):
    response: ResponseEnvelope


class RenderInput(StrictModel):
    query: str = ""
    answer: str
    response_type: ResponseType
    language: Literal["zh", "en"] = "zh"
    trace_id: str = ""
    intent: str = ""
    knowledge_points: list[str] = Field(default_factory=list)
    sources: list[PublicSource] = Field(default_factory=list)
    validation_passed: StrictBool
    conversation_history: list[PublicConversationTurn] = Field(default_factory=list)
    conversation_summary: str = ""
    exercise_state: PublicExerciseState | None = None
    clarification: PublicClarification | None = None
    metrics: PublicMetrics = Field(default_factory=PublicMetrics)
    cached: bool = False


class AnswerEnvelope(ResponseEnvelope):
    language: Literal["zh", "en"] = "zh"

