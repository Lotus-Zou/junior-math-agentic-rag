from __future__ import annotations

import pytest

from agentic_rag.exercises.generator import (
    AdaptiveExerciseGenerator,
    ExerciseGenerationError,
    prompt_fingerprint,
)
from agentic_rag.exercises.models import ExerciseRequest, ExerciseSessionState
from agentic_rag.exercises.store import ExerciseStore
from agentic_rag.exercises.progression import (
    next_difficulty,
    parse_practice_preferences,
    update_mastery,
)
from agentic_rag.exercises.validation import validate_generated_exercise


def test_recent_twenty_fingerprints_are_avoided():
    generator = AdaptiveExerciseGenerator(max_attempts=100)
    recent = [
        generator.generate(ExerciseRequest(topic="geometry", seed=index)).fingerprint
        for index in range(20)
    ]
    item = generator.generate(
        ExerciseRequest(
            topic="geometry",
            recent_fingerprints=recent,
            seed=100,
        )
    )

    assert item.fingerprint not in recent


@pytest.mark.parametrize(
    ("current", "outcome", "delta", "expected"),
    [
        (2, "correct", 0, 3),
        (3, "correct_after_hint", 0, 3),
        (3, "incorrect", 0, 2),
        (5, "correct", 0, 5),
        (1, "incorrect", 0, 1),
        (2, "unknown", 1, 3),
        (4, "correct", 1, 5),
    ],
)
def test_progression(current, outcome, delta, expected):
    assert next_difficulty(current, outcome, delta) == expected


def test_same_topic_produces_varied_verified_items():
    items = [
        AdaptiveExerciseGenerator().generate(
            ExerciseRequest(topic="geometry", seed=index)
        )
        for index in range(12)
    ]

    assert len({item.fingerprint for item in items}) >= 10
    assert all(validate_generated_exercise(item).passed for item in items)


def test_composite_preference_request_is_parsed():
    request = parse_practice_preferences("八年级几何难一点", current=None)

    assert (request.grade, request.topic, request.difficulty) == (8, "geometry", 3)


@pytest.mark.parametrize(
    "query",
    [
        "帮我出一道九年级的数学竞赛题",
        "[初中9年级] 生成一道竞赛题",
    ],
)
def test_grade_nine_competition_request_generates_verified_challenge(query):
    request = parse_practice_preferences(query, current=None)

    assert (
        request.grade,
        request.topic,
        request.difficulty,
        request.exercise_type,
    ) == (9, "algebra", 5, "mixed")
    item = AdaptiveExerciseGenerator().generate(
        request.model_copy(update={"seed": 29})
    )
    assert item.template_id == "alg.consecutive_squares.v1"
    assert validate_generated_exercise(item).passed is True
    assert item.answer_signature not in item.problem


def test_explicit_easy_function_request_overrides_history():
    current = ExerciseSessionState(
        session_id="s1",
        current_exercise_id="e1",
        current_topic="geometry",
        current_grade=8,
        current_difficulty=4,
        current_exercise_type="proof",
        mastery={"一次函数": 0.9},
    )

    request = parse_practice_preferences("来一道简单的一次函数题", current=current)

    assert request.topic == "linear_function"
    assert request.difficulty == 1
    assert request.grade == 8
    assert request.exercise_type == "calculation"


def test_omitted_preferences_inherit_private_session_defaults():
    current = ExerciseSessionState(
        session_id="s1",
        current_exercise_id="e1",
        current_topic="algebra",
        current_grade=7,
        current_difficulty=2,
        current_exercise_type="calculation",
    )

    request = parse_practice_preferences("再来一道", current=current)

    assert (
        request.topic,
        request.grade,
        request.difficulty,
        request.exercise_type,
    ) == ("algebra", 7, 2, "calculation")


def test_english_composite_preferences_are_supported():
    request = parse_practice_preferences(
        "Give me a harder grade 8 geometry proof", current=None
    )

    assert (
        request.topic,
        request.grade,
        request.difficulty,
        request.exercise_type,
    ) == ("geometry", 8, 3, "proof")
    assert request.language == "en"


def test_english_request_generates_english_problem():
    item = AdaptiveExerciseGenerator().generate(
        ExerciseRequest(topic="algebra", grade=7, difficulty=2, language="en", seed=8)
    )

    assert item.problem.startswith("Solve the equation")
    assert validate_generated_exercise(item).passed


def test_store_and_generator_share_prompt_fingerprint_definition():
    item = AdaptiveExerciseGenerator().generate(
        ExerciseRequest(topic="geometry", seed=3)
    )
    store = ExerciseStore(ttl_seconds=60)
    public = store.start(item, mastery={})

    session = store.get_session(public.session_id)

    assert session.recent_prompt_fingerprints == [
        prompt_fingerprint(item.problem, item.hint)
    ]


def test_candidate_generation_error_is_retried_within_budget(monkeypatch):
    import agentic_rag.exercises.generator as module

    original = module.generate_from_template
    calls = 0

    def flaky(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("invalid sampled candidate")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "generate_from_template", flaky)

    item = AdaptiveExerciseGenerator(max_attempts=2).generate(
        ExerciseRequest(topic="algebra", grade=7, difficulty=2, seed=8)
    )

    assert calls == 2
    assert validate_generated_exercise(item).passed


def test_generator_relaxes_only_answer_signature_diversity():
    generator = AdaptiveExerciseGenerator(max_attempts=20)
    baseline = generator.generate(
        ExerciseRequest(topic="algebra", grade=7, difficulty=1, seed=5)
    )
    item = generator.generate(
        ExerciseRequest(
            topic="algebra",
            grade=7,
            difficulty=1,
            recent_answer_signatures=[baseline.answer_signature],
            seed=5,
        )
    )

    assert validate_generated_exercise(item).passed
    assert item.fingerprint != ""


def test_generator_never_relaxes_exact_problem_fingerprint():
    generator = AdaptiveExerciseGenerator(max_attempts=1)
    baseline = generator.generate(
        ExerciseRequest(topic="algebra", grade=7, difficulty=1, seed=5)
    )

    with pytest.raises(ExerciseGenerationError):
        generator.generate(
            ExerciseRequest(
                topic="algebra",
                grade=7,
                difficulty=1,
                recent_fingerprints=[baseline.fingerprint],
                seed=5,
            )
        )


def test_unsupported_filter_combination_fails_explicitly():
    with pytest.raises(ExerciseGenerationError, match="No template"):
        AdaptiveExerciseGenerator().generate(
            ExerciseRequest(
                topic="geometry",
                grade=7,
                difficulty=4,
                exercise_type="proof",
            )
        )


def test_mastery_update_is_bounded_and_outcome_sensitive():
    mastery = {"一次函数": 0.5, "斜率": 0.95}

    correct = update_mastery(mastery, ["一次函数", "斜率"], "correct")
    incorrect = update_mastery(mastery, ["一次函数"], "incorrect")

    assert correct["一次函数"] == pytest.approx(0.6)
    assert correct["斜率"] == pytest.approx(0.96)
    assert incorrect["一次函数"] == pytest.approx(0.4)
    assert all(0.0 <= value <= 1.0 for value in [*correct.values(), *incorrect.values()])
