from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import agentic_rag.fast_path as fast_path
from agentic_rag.exercises.checking import check_exercise_answer
from agentic_rag.exercises.generator import AdaptiveExerciseGenerator
from agentic_rag.exercises.store import ExerciseStore
from agentic_rag.exercises.templates import generate_from_template
from app import app


@pytest.fixture
def isolated_runtime(monkeypatch):
    store = ExerciseStore(ttl_seconds=60)
    monkeypatch.setattr(fast_path, "_adaptive_exercise_store", store)
    monkeypatch.setattr(
        fast_path, "_adaptive_exercise_generator", AdaptiveExerciseGenerator()
    )
    return store


def test_next_exercise_does_not_repeat(isolated_runtime):
    first = fast_path.build_fast_response("几何", [], language="zh")
    second = fast_path.build_fast_response(
        "再来一道",
        first["conversation_history"],
        language="zh",
        exercise_state=first["exercise_state"],
    )

    assert first["response_type"] == second["response_type"] == "guided_exercise"
    assert second["exercise_state"]["fingerprint"] != first["exercise_state"]["fingerprint"]
    assert second["exercise_state"]["session_id"] == first["exercise_state"]["session_id"]


def test_harder_increases_level_without_answer_leak(isolated_runtime):
    first = fast_path.build_fast_response("一次函数", [], language="zh")
    second = fast_path.build_fast_response(
        "难一点",
        first["conversation_history"],
        language="zh",
        exercise_state=first["exercise_state"],
    )

    assert second["exercise_state"]["difficulty"] == min(
        first["exercise_state"]["difficulty"] + 1, 5
    )
    serialized = str(second["exercise_state"])
    assert "solution" not in serialized
    assert "answer_signature" not in serialized
    assert "parameters" not in serialized


def test_correct_answer_is_verified_by_opaque_exercise_id(isolated_runtime):
    exercise = generate_from_template("alg.linear_equation.v1", 2, 7, seed=9)
    public = isolated_runtime.start(exercise, mastery={})

    result = fast_path.build_fast_response(
        f"x = {exercise.parameters['x']}",
        [],
        language="zh",
        exercise_state=public.model_dump(),
    )

    assert result["response_type"] == "verified_answer"
    assert result["validation_passed"] is True
    assert exercise.solution in result["answer"]
    assert result["exercise_state"]["exercise_id"] == exercise.exercise_id


def test_wrong_answer_keeps_solution_hidden(isolated_runtime):
    exercise = generate_from_template("alg.linear_equation.v1", 2, 7, seed=9)
    public = isolated_runtime.start(exercise, mastery={})

    result = fast_path.build_fast_response(
        "x = 999",
        [],
        language="zh",
        exercise_state=public.model_dump(),
    )

    assert result["response_type"] == "guided_exercise"
    assert exercise.solution not in result["answer"]
    assert exercise.answer_signature not in result["answer"]


def test_tampered_public_state_cannot_select_private_answer(isolated_runtime):
    exercise = generate_from_template("alg.linear_equation.v1", 2, 7, seed=9)
    public = isolated_runtime.start(exercise, mastery={}).model_dump()
    tampered = {**public, "fingerprint": "0" * 64}

    result = fast_path.build_fast_response(
        f"x = {exercise.parameters['x']}",
        [],
        language="zh",
        exercise_state=tampered,
    )

    assert result["response_type"] == "clarification_required"
    assert exercise.solution not in result["answer"]


def test_expired_exercise_requests_a_fresh_problem():
    result = check_exercise_answer(ExerciseStore(ttl_seconds=60), "missing-id", "70度")

    assert result.status == "expired"
    assert result.passed is False


def test_api_does_not_reuse_cached_adaptive_exercise(monkeypatch, isolated_runtime):
    import app as api

    monkeypatch.setattr(api.answer_cache, "get", lambda _payload: None)
    cache_writes = []
    monkeypatch.setattr(api.answer_cache, "set", lambda *args: cache_writes.append(args))

    with TestClient(app, raise_server_exceptions=False) as client:
        first = client.post("/ask", json={"query": "几何", "language": "zh"}).json()
        second = client.post(
            "/ask",
            json={
                "query": "几何",
                "language": "zh",
                "conversation_history": first["conversation_history"],
                "exercise_state": first["exercise_state"],
            },
        ).json()

    assert first["exercise_state"]["exercise_id"] != second["exercise_state"]["exercise_id"]
    assert first["exercise_state"]["fingerprint"] != second["exercise_state"]["fingerprint"]
    assert cache_writes == []


def test_api_round_trips_public_exercise_state(monkeypatch, isolated_runtime):
    import app as api

    monkeypatch.setattr(api.answer_cache, "get", lambda _payload: None)
    monkeypatch.setattr(api.answer_cache, "set", lambda *_args: None)

    with TestClient(app, raise_server_exceptions=False) as client:
        first = client.post("/ask", json={"query": "代数", "language": "zh"}).json()
        second = client.post(
            "/ask",
            json={
                "query": "再来一道",
                "language": "zh",
                "conversation_history": first["conversation_history"],
                "exercise_state": first["exercise_state"],
            },
        ).json()

    assert second["response_type"] == "guided_exercise"
    assert second["exercise_state"]["session_id"] == first["exercise_state"]["session_id"]


def test_frontend_sends_and_clears_only_public_exercise_state():
    source = open("static/app.js", encoding="utf-8").read()

    assert "exercise_state: state.exercise" in source
    assert 'state.exercise = result.exercise_state' in source
    assert "state.exercise = null" in source
    for forbidden in ("solution", "answer_signature", "parameters"):
        assert f"state.exercise.{forbidden}" not in source
