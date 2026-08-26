"""Core junior-mathematics domain objects."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ConversationTurn(DomainModel):
    role: Literal["student", "tutor", "system"]
    content: str = Field(min_length=1, max_length=12000)


class Question(DomainModel):
    text: str = Field(min_length=1, max_length=12000)
    language: Literal["zh", "en"] = "zh"
    grade: str | None = None
    chapter: str | None = None
    knowledge_points: list[str] = Field(default_factory=list)


class StudentAttempt(DomainModel):
    answer: str = Field(default="", max_length=12000)
    steps: list[str] = Field(default_factory=list)

