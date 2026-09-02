import json
from datetime import datetime, timedelta, timezone

from langchain_core.messages import AIMessage

from agentic_rag.domain.schemas import ResponseEnvelope
from agentic_rag.fast_path import build_fast_response
from agentic_rag.skill_runtime.contracts import SkillContext
from agentic_rag.skill_runtime.executor import SkillExecutor
from agentic_rag.skill_runtime.pipeline import PipelineExecutor, PipelineLoader
from agentic_rag.skill_runtime.registry import get_default_registry
from agentic_rag.tutor_agent import enrich_curriculum_response


def _baseline() -> ResponseEnvelope:
    response = build_fast_response("解方程 2x+3=11", [], language="zh")
    return ResponseEnvelope.model_validate(response)


def _patch_author_and_critic(monkeypatch, answer):
    from agentic_rag import chains

    calls = []

    def fake_invoke(instance, _messages):
        calls.append(instance)
        if instance is chains.generator_llm:
            return AIMessage(content=answer)
        return AIMessage(content=json.dumps({"passed": True, "issues": []}))

    monkeypatch.setattr(type(chains.generator_llm), "invoke", fake_invoke)
    return calls


def test_tutor_author_and_independent_critic_are_both_called(monkeypatch):
    baseline = _baseline()
    calls = _patch_author_and_critic(monkeypatch, baseline.answer)

    result = enrich_curriculum_response(
        query="解方程 2x+3=11",
        baseline=baseline,
        language="zh",
        enabled=True,
    )

    from agentic_rag import chains

    assert calls == [chains.generator_llm, chains.critic_llm]
    assert result.answer == baseline.answer
    assert result.validation_passed is True
    assert result.metrics.tool_calls == (baseline.metrics.tool_calls or 0) + 2


def test_tutor_falls_back_to_verified_baseline_when_math_changes(monkeypatch):
    baseline = _baseline()
    calls = _patch_author_and_critic(monkeypatch, baseline.answer.replace("x = 4", "x = 7"))

    result = enrich_curriculum_response(
        query="解方程 2x+3=11",
        baseline=baseline,
        language="zh",
        enabled=True,
    )

    assert len(calls) == 2
    assert result.answer == baseline.answer
    assert result.metrics.tool_calls == (baseline.metrics.tool_calls or 0) + 2


def test_deterministic_pipeline_passes_through_tutor_skill(monkeypatch):
    baseline = _baseline()
    calls = _patch_author_and_critic(monkeypatch, baseline.answer)
    registry = get_default_registry()
    pipeline = PipelineLoader(registry).load(
        "agentic_rag/pipelines/correction.yaml"
    )
    context = SkillContext(
        request_id="tutor-pipeline",
        trace_id="tutor-pipeline",
        deadline_at=datetime.now(timezone.utc) + timedelta(seconds=10),
        feature_flags={"tutor_agent": True, "exercise_agent": False},
        policy_set={"allow:math.exercise_generate"},
    )

    state = PipelineExecutor(SkillExecutor(registry)).run(
        pipeline,
        {"query": "解方程 2x+3=11", "language": "zh"},
        context,
    )

    assert state["turn_router"].route == "deterministic"
    assert state["curriculum_tutor"].response.response_type == "verified_answer"
    assert state["response_render"].metrics.tool_calls == 2
    assert len(calls) == 2
