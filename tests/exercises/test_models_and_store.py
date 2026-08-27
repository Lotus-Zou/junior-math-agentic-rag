from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

import app as api
from agentic_rag.exercises.models import GeneratedExercise, PublicExerciseState
from agentic_rag.exercises.store import ExerciseStore


class FakeClock:
    def __init__(self, value: float):
        self.value = value

    def __call__(self) -> float:
        return self.value


def make_exercise(**overrides) -> GeneratedExercise:
    values = {
        "exercise_id": "ex-1",
        "topic": "geometry",
        "grade": 8,
        "difficulty": 2,
        "exercise_type": "calculation",
        "template_id": "geo.isosceles.base_angles.v1",
        "problem": "顶角为 40 度，求两个底角。",
        "hint": "使用三角形内角和与等腰三角形性质。",
        "solution": "两个底角都是 70 度。",
        "answer_signature": "70|70",
        "knowledge_points": ["等腰三角形", "三角形内角和"],
        "parameters": {"vertex_angle": 40, "base_angle": 70},
        "fingerprint": "problem-fp-1",
    }
    return GeneratedExercise(**(values | overrides))


def test_public_state_never_serializes_hidden_solution():
    exercise = make_exercise(solution="两个底角都是 70 度", answer_signature="70|70")
    public = ExerciseStore(ttl_seconds=60).start(exercise, mastery={})

    text = public.model_dump_json()

    assert "70|70" not in text
    assert "solution" not in text
    assert "parameters" not in text
    assert "answer_signature" not in text


def test_store_expires_exercise_and_session_together():
    clock = FakeClock(100.0)
    store = ExerciseStore(ttl_seconds=30, clock=clock)
    public = store.start(make_exercise(), mastery={"等腰三角形": 0.5})

    clock.value = 131.0

    assert store.get_exercise("ex-1") is None
    assert store.get_session(public.session_id) is None


def test_store_returns_defensive_copies():
    store = ExerciseStore(ttl_seconds=60)
    public = store.start(make_exercise(), mastery={"等腰三角形": 0.5})

    exercise = store.get_exercise(public.exercise_id)
    session = store.get_session(public.session_id)
    exercise.parameters["base_angle"] = 1
    session.mastery["等腰三角形"] = 1.0

    assert store.get_exercise(public.exercise_id).parameters["base_angle"] == 70
    assert store.get_session(public.session_id).mastery["等腰三角形"] == 0.5


def test_active_exercise_id_cannot_be_overwritten():
    store = ExerciseStore(ttl_seconds=60)
    store.start(make_exercise(), mastery={})

    with pytest.raises(ValueError, match="already active"):
        store.start(
            make_exercise(problem="同一个 ID 下的另一道题", solution="另一个答案"),
            mastery={},
        )

    assert store.get_exercise("ex-1").problem == "顶角为 40 度，求两个底角。"


def test_start_updates_bounded_private_session_history():
    store = ExerciseStore(ttl_seconds=60)
    public = None
    for index in range(25):
        public = store.start(
            make_exercise(
                exercise_id=f"ex-{index}",
                problem=f"第 {index} 题",
                fingerprint=f"problem-fp-{index}",
                answer_signature=f"answer-{index}",
            ),
            mastery={"等腰三角形": 0.5},
            session_id=public.session_id if public else None,
        )

    session = store.get_session(public.session_id)
    assert len(session.recent_fingerprints) == 20
    assert len(session.recent_prompt_fingerprints) == 10
    assert len(session.recent_answer_signatures) == 5
    assert session.current_exercise_id == "ex-24"


def test_store_is_safe_under_parallel_reads_and_writes():
    store = ExerciseStore(ttl_seconds=60)

    def create(index: int) -> str:
        public = store.start(
            make_exercise(
                exercise_id=f"parallel-{index}",
                problem=f"并发题目 {index}",
                fingerprint=f"parallel-fp-{index}",
            ),
            mastery={},
        )
        assert store.get_exercise(public.exercise_id) is not None
        return public.session_id

    with ThreadPoolExecutor(max_workers=8) as pool:
        session_ids = list(pool.map(create, range(40)))

    assert len(set(session_ids)) == 40


def test_ask_request_accepts_only_public_exercise_state():
    public = ExerciseStore(ttl_seconds=60).start(make_exercise(), mastery={})
    request = api.AskRequest(query="再来一道", exercise_state=public.model_dump())

    assert isinstance(request.exercise_state, PublicExerciseState)
    assert request.model_dump()["exercise_state"]["exercise_id"] == "ex-1"

    with pytest.raises(ValidationError):
        api.AskRequest(
            query="再来一道",
            exercise_state={**public.model_dump(), "solution": "两个底角都是 70 度"},
        )


@pytest.mark.parametrize("field", ["solution", "answer_signature", "parameters"])
def test_public_state_rejects_private_fields(field):
    public = ExerciseStore(ttl_seconds=60).start(make_exercise(), mastery={}).model_dump()

    with pytest.raises(ValidationError):
        PublicExerciseState.model_validate({**public, field: "secret"})
