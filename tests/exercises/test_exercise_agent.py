import json

from langchain_core.messages import AIMessage

from agentic_rag.exercises.agent import enhance_verified_exercise
from agentic_rag.exercises.agent import ExerciseAgentResult
from agentic_rag.exercises.generator import AdaptiveExerciseGenerator
from agentic_rag.exercises.models import ExerciseRequest
from agentic_rag.exercises.validation import validate_generated_exercise
from agentic_rag.fast_path import build_agentic_exercise_response


def test_exercise_author_and_independent_critic_both_run(monkeypatch):
    request = ExerciseRequest(
        topic="geometry",
        grade=8,
        difficulty=3,
        exercise_type="mixed",
        seed=17,
    )
    seed = AdaptiveExerciseGenerator().generate(request)
    from agentic_rag import chains

    calls = []

    def fake_invoke(instance, _messages):
        calls.append(instance)
        if instance is chains.exercise_llm:
            return AIMessage(content=json.dumps({
                "problem": seed.problem,
                "hint": seed.hint,
                "solution": seed.solution,
            }, ensure_ascii=False))
        return AIMessage(content='{"passed":true,"issues":[]}')

    monkeypatch.setattr(type(chains.exercise_llm), "invoke", fake_invoke)
    result = enhance_verified_exercise(seed, request)

    assert result.tool_calls == 2
    assert result.author_passed is True
    assert result.critic_passed is True
    assert validate_generated_exercise(result.exercise).passed is True
    assert calls == [chains.exercise_llm, chains.critic_llm]


def test_agent_verified_exercise_satisfies_public_response_contract(monkeypatch):
    from agentic_rag.exercises import agent

    def verified(seed, _request):
        return ExerciseAgentResult(
            exercise=seed,
            tool_calls=2,
            author_passed=True,
            critic_passed=True,
        )

    monkeypatch.setattr(agent, "enhance_verified_exercise", verified)
    response = build_agentic_exercise_response(
        "几何",
        [],
        "",
        "zh",
        agent_enabled=True,
    )

    assert response is not None
    assert response["response_type"] == "guided_exercise"
    assert response["validation_passed"] is True
    assert response["metrics"]["tool_calls"] == 2


def test_failed_author_attempt_is_visible_in_tool_metrics(monkeypatch):
    request = ExerciseRequest(
        topic="geometry",
        grade=8,
        difficulty=3,
        exercise_type="mixed",
        seed=23,
    )
    seed = AdaptiveExerciseGenerator().generate(request)
    from agentic_rag import chains

    def fail_author(_instance, _messages):
        raise TimeoutError("provider timeout")

    monkeypatch.setattr(type(chains.exercise_llm), "invoke", fail_author)
    result = enhance_verified_exercise(seed, request)

    assert result.tool_calls == 0
    assert result.model_attempts == 1
    assert result.model_failures == 1
    assert result.exercise == seed
    assert result.author_passed is False
    assert result.critic_passed is False
