"""Strict domain models for generated exercises and public session state."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ExerciseTopic = Literal["geometry", "algebra", "linear_function"]
ExerciseType = Literal["calculation", "proof", "application", "mixed"]


class StrictExerciseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


class ExerciseRequest(StrictExerciseModel):
    topic: ExerciseTopic
    grade: int = Field(default=8, ge=7, le=9)
    difficulty: int = Field(default=2, ge=1, le=5)
    exercise_type: ExerciseType = "calculation"
    recent_fingerprints: list[str] = Field(default_factory=list, max_length=20)
    recent_answer_signatures: list[str] = Field(default_factory=list, max_length=10)
    seed: int | None = None


class GeneratedExercise(StrictExerciseModel):
    """Private server-side exercise record. Never return this model to a client."""

    exercise_id: str = Field(min_length=1, max_length=128)
    topic: ExerciseTopic
    grade: int = Field(ge=7, le=9)
    difficulty: int = Field(ge=1, le=5)
    exercise_type: ExerciseType
    template_id: str = Field(min_length=1, max_length=160)
    problem: str = Field(min_length=1, max_length=8000)
    hint: str = Field(min_length=1, max_length=4000)
    solution: str = Field(min_length=1, max_length=12000)
    answer_signature: str = Field(min_length=1, max_length=1000)
    knowledge_points: list[str] = Field(min_length=1, max_length=20)
    parameters: dict[str, Any]
    fingerprint: str = Field(min_length=1, max_length=256)

    @field_validator("knowledge_points")
    @classmethod
    def unique_knowledge_points(cls, value: list[str]) -> list[str]:
        if any(not item for item in value):
            raise ValueError("knowledge_points cannot contain empty values")
        if len(set(value)) != len(value):
            raise ValueError("knowledge_points must be unique")
        return value


class PublicExerciseState(StrictExerciseModel):
    """Opaque browser-safe pointer to private exercise and session records."""

    exercise_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    topic: ExerciseTopic
    grade: int = Field(ge=7, le=9)
    difficulty: int = Field(ge=1, le=5)
    exercise_type: ExerciseType
    template_id: str = Field(min_length=1, max_length=160)
    fingerprint: str = Field(min_length=1, max_length=256)
    knowledge_points: list[str] = Field(min_length=1, max_length=20)


class ExerciseSessionState(StrictExerciseModel):
    """Private adaptive state retained only by the server-side store."""

    session_id: str = Field(min_length=1, max_length=128)
    current_exercise_id: str = Field(min_length=1, max_length=128)
    recent_fingerprints: list[str] = Field(default_factory=list, max_length=20)
    recent_prompt_fingerprints: list[str] = Field(default_factory=list, max_length=10)
    recent_answer_signatures: list[str] = Field(default_factory=list, max_length=5)
    mastery: dict[str, float] = Field(default_factory=dict)

    @field_validator("mastery")
    @classmethod
    def mastery_scores_are_probabilities(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not key or score < 0.0 or score > 1.0 for key, score in value.items()):
            raise ValueError("mastery values must be between 0 and 1")
        return value
