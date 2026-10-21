"""Bounded adaptive selection over independently verified exercise templates."""

from __future__ import annotations

import hashlib
import random
import unicodedata

from agentic_rag.exercises.models import ExerciseRequest, GeneratedExercise
from agentic_rag.exercises.templates import TEMPLATE_REGISTRY, generate_from_template
from agentic_rag.exercises.validation import validate_generated_exercise


class ExerciseGenerationError(RuntimeError):
    pass


def prompt_fingerprint(problem: str, hint: str) -> str:
    normalized = unicodedata.normalize("NFKC", f"{problem}\n{hint}").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class AdaptiveExerciseGenerator:
    def __init__(self, max_attempts: int = 100) -> None:
        if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        self._max_attempts = max_attempts

    @staticmethod
    def _definitions(request: ExerciseRequest):
        return [
            definition
            for definition in TEMPLATE_REGISTRY.values()
            if definition.topic == request.topic
            and request.grade in definition.grades
            and request.difficulty in definition.difficulties
            and (
                request.exercise_type == "mixed"
                or definition.exercise_type == request.exercise_type
            )
        ]

    def generate(self, request: ExerciseRequest) -> GeneratedExercise:
        request = ExerciseRequest.model_validate(request)
        definitions = self._definitions(request)
        if not definitions:
            raise ExerciseGenerationError(
                "No template matches the requested topic, grade, difficulty, and exercise type"
            )

        recent_fingerprints = set(request.recent_fingerprints)
        recent_prompts = set(request.recent_prompt_fingerprints)
        recent_answers = set(request.recent_answer_signatures)
        rng = random.Random(request.seed)
        answer_diversity_fallback: GeneratedExercise | None = None

        for _ in range(self._max_attempts):
            definition = rng.choice(definitions)
            candidate_seed = rng.getrandbits(64)
            try:
                candidate = generate_from_template(
                    definition.template_id,
                    request.difficulty,
                    request.grade,
                    candidate_seed,
                    language=request.language,
                )
            except ValueError:
                continue
            validation = validate_generated_exercise(candidate)
            if not validation.passed:
                continue
            if candidate.fingerprint in recent_fingerprints:
                continue
            if prompt_fingerprint(candidate.problem, candidate.hint) in recent_prompts:
                continue
            if candidate.answer_signature in recent_answers:
                if answer_diversity_fallback is None:
                    answer_diversity_fallback = candidate
                continue
            return candidate

        if answer_diversity_fallback is not None:
            return answer_diversity_fallback
        raise ExerciseGenerationError(
            "No verified non-repeating exercise was found within the attempt budget"
        )
