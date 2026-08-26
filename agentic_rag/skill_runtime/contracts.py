"""Execution contracts with safe, serializable provenance and artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.domain.models import ConversationTurn, Question, StudentAttempt

T = TypeVar("T")


class RuntimeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class SkillStatus(str, Enum):
    OK = "OK"
    CLARIFY = "CLARIFY"
    UNSUPPORTED = "UNSUPPORTED"
    RETRYABLE_ERROR = "RETRYABLE_ERROR"
    FATAL_ERROR = "FATAL_ERROR"


class Artifact(RuntimeModel):
    artifact_id: str
    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)


class Citation(RuntimeModel):
    source_id: str
    label: str = ""
    excerpt: str = ""


class SkillContext(RuntimeModel):
    request_id: str
    trace_id: str
    session_id: str = "anonymous"
    tenant_id: str = "default"
    language: str = "zh"
    deadline_at: datetime
    question: Question | None = None
    student_attempt: StudentAttempt | None = None
    conversation: list[ConversationTurn] = Field(default_factory=list)
    policy_set: set[str] = Field(default_factory=set)
    feature_flags: dict[str, bool] = Field(default_factory=dict)
    model_profile: str = "default"

    @property
    def remaining_budget_ms(self) -> int:
        delta = self.deadline_at - datetime.now(timezone.utc)
        return max(0, int(delta.total_seconds() * 1000))


class SkillResult(RuntimeModel, Generic[T]):
    status: SkillStatus
    value: T | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, str] = Field(default_factory=dict)
    safe_error: str = ""

