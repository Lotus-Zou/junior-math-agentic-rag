"""Thread-safe in-memory storage for private exercises and adaptive sessions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from threading import Lock
import time
from typing import Callable, Generic, TypeVar
import unicodedata
import uuid

from agentic_rag.exercises.models import (
    ExerciseSessionState,
    GeneratedExercise,
    PublicExerciseState,
)


T = TypeVar("T")


@dataclass(frozen=True)
class _ExpiringRecord(Generic[T]):
    value: T
    expires_at: float


def _append_bounded(values: list[str], value: str, limit: int) -> list[str]:
    return [*values, value][-limit:]


def _prompt_fingerprint(problem: str, hint: str) -> str:
    normalized = unicodedata.normalize("NFKC", f"{problem}\n{hint}").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class ExerciseStore:
    """Own private exercise records; expose only validated public pointers."""

    def __init__(
        self,
        ttl_seconds: float = 1800,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a positive finite number")
        self._ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._lock = Lock()
        self._exercises: dict[str, _ExpiringRecord[GeneratedExercise]] = {}
        self._sessions: dict[str, _ExpiringRecord[ExerciseSessionState]] = {}

    def _now(self) -> float:
        value = float(self._clock())
        if not math.isfinite(value):
            raise RuntimeError("exercise store clock returned a non-finite value")
        return value

    def _prune_locked(self, now: float) -> None:
        self._exercises = {
            key: record
            for key, record in self._exercises.items()
            if record.expires_at > now
        }
        self._sessions = {
            key: record
            for key, record in self._sessions.items()
            if record.expires_at > now
        }

    def start(
        self,
        exercise: GeneratedExercise,
        mastery: dict[str, float],
        *,
        session_id: str | None = None,
    ) -> PublicExerciseState:
        private_exercise = GeneratedExercise.model_validate(exercise).model_copy(deep=True)
        now = self._now()
        expires_at = now + self._ttl_seconds
        with self._lock:
            self._prune_locked(now)
            if private_exercise.exercise_id in self._exercises:
                raise ValueError(
                    f"exercise_id is already active: {private_exercise.exercise_id}"
                )
            prior_record = self._sessions.get(session_id or "")
            prior = prior_record.value if prior_record is not None else None
            resolved_session_id = prior.session_id if prior is not None else uuid.uuid4().hex
            merged_mastery = dict(prior.mastery) if prior is not None else {}
            merged_mastery.update(mastery)
            session = ExerciseSessionState(
                session_id=resolved_session_id,
                current_exercise_id=private_exercise.exercise_id,
                current_topic=private_exercise.topic,
                current_grade=private_exercise.grade,
                current_difficulty=private_exercise.difficulty,
                current_exercise_type=private_exercise.exercise_type,
                recent_fingerprints=_append_bounded(
                    prior.recent_fingerprints if prior else [],
                    private_exercise.fingerprint,
                    20,
                ),
                recent_prompt_fingerprints=_append_bounded(
                    prior.recent_prompt_fingerprints if prior else [],
                    _prompt_fingerprint(private_exercise.problem, private_exercise.hint),
                    10,
                ),
                recent_answer_signatures=_append_bounded(
                    prior.recent_answer_signatures if prior else [],
                    private_exercise.answer_signature,
                    5,
                ),
                mastery=merged_mastery,
            )
            self._exercises[private_exercise.exercise_id] = _ExpiringRecord(
                value=private_exercise,
                expires_at=expires_at,
            )
            self._sessions[resolved_session_id] = _ExpiringRecord(
                value=session,
                expires_at=expires_at,
            )

        return PublicExerciseState(
            exercise_id=private_exercise.exercise_id,
            session_id=resolved_session_id,
            topic=private_exercise.topic,
            grade=private_exercise.grade,
            difficulty=private_exercise.difficulty,
            exercise_type=private_exercise.exercise_type,
            template_id=private_exercise.template_id,
            fingerprint=private_exercise.fingerprint,
            knowledge_points=list(private_exercise.knowledge_points),
        )

    def get_exercise(self, exercise_id: str) -> GeneratedExercise | None:
        now = self._now()
        with self._lock:
            self._prune_locked(now)
            record = self._exercises.get(exercise_id)
            return record.value.model_copy(deep=True) if record is not None else None

    def get_session(self, session_id: str) -> ExerciseSessionState | None:
        now = self._now()
        with self._lock:
            self._prune_locked(now)
            record = self._sessions.get(session_id)
            return record.value.model_copy(deep=True) if record is not None else None

    def record_outcome(
        self,
        session_id: str,
        exercise_id: str,
        outcome: str,
    ) -> ExerciseSessionState | None:
        from agentic_rag.exercises.progression import update_mastery

        now = self._now()
        with self._lock:
            self._prune_locked(now)
            session_record = self._sessions.get(session_id)
            exercise_record = self._exercises.get(exercise_id)
            if session_record is None or exercise_record is None:
                return None
            session = session_record.value
            exercise = exercise_record.value
            if session.current_exercise_id != exercise_id:
                return None
            updated = session.model_copy(
                update={
                    "mastery": update_mastery(
                        session.mastery,
                        exercise.knowledge_points,
                        outcome,
                    )
                },
                deep=True,
            )
            expires_at = now + self._ttl_seconds
            self._sessions[session_id] = _ExpiringRecord(updated, expires_at)
            self._exercises[exercise_id] = _ExpiringRecord(exercise, expires_at)
            return updated.model_copy(deep=True)
