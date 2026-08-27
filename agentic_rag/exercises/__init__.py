"""Verified exercise generation and private session state."""

from agentic_rag.exercises.models import (
    ExerciseRequest,
    ExerciseSessionState,
    GeneratedExercise,
    PublicExerciseState,
)
from agentic_rag.exercises.generator import AdaptiveExerciseGenerator, ExerciseGenerationError
from agentic_rag.exercises.progression import (
    next_difficulty,
    parse_practice_preferences,
    update_mastery,
)
from agentic_rag.exercises.store import ExerciseStore
from agentic_rag.exercises.templates import TEMPLATE_REGISTRY, generate_from_template
from agentic_rag.exercises.validation import validate_generated_exercise

__all__ = [
    "ExerciseRequest",
    "ExerciseSessionState",
    "ExerciseStore",
    "ExerciseGenerationError",
    "GeneratedExercise",
    "PublicExerciseState",
    "TEMPLATE_REGISTRY",
    "AdaptiveExerciseGenerator",
    "generate_from_template",
    "next_difficulty",
    "parse_practice_preferences",
    "update_mastery",
    "validate_generated_exercise",
]
