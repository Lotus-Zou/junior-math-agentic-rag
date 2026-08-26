from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agentic_rag.skill_runtime.contracts import SkillContext, SkillStatus
from agentic_rag.skill_runtime.errors import ManifestError, PipelineError
from agentic_rag.skill_runtime.executor import SkillExecutor
from agentic_rag.skill_runtime.pipeline import PipelineExecutor, PipelineLoader
from agentic_rag.skill_runtime.manifest import SkillManifest
from agentic_rag.skill_runtime.registry import SkillRegistry, get_default_registry

ROOT = Path(__file__).resolve().parents[2]


def context():
    return SkillContext(request_id="request-1", trace_id="trace-1", deadline_at=datetime.now(timezone.utc) + timedelta(seconds=8))


def test_registry_discovers_versioned_skills():
    registry = get_default_registry()
    assert len(registry.list()) == 17
    assert registry.resolve("math.input_guard@1").version == "1.0.0"


def test_registry_rejects_duplicate_version():
    manifest = get_default_registry().resolve("math.input_guard@1")
    registry = SkillRegistry()
    registry.register(manifest)
    with pytest.raises(ManifestError):
        registry.register(manifest)


def test_executor_validates_input_and_emits_hashed_trace():
    executor = SkillExecutor(get_default_registry())
    result = executor.execute("math.input_guard@1", {"query": "  解方程 2x+3=11  "}, context())
    assert result.status == SkillStatus.OK
    assert result.value.normalized_query == "解方程 2x+3=11"
    event = executor.telemetry.events[-1]
    assert len(event.input_hash) == 64
    assert "解方程" not in event.model_dump_json()


def test_executor_rejects_extra_fields_without_leaking_details():
    result = SkillExecutor(get_default_registry()).execute("math.input_guard@1", {"query": "x=1", "secret": "never-log-me"}, context())
    assert result.status == SkillStatus.FATAL_ERROR
    assert result.safe_error == "请求参数格式不正确，请检查后重试。"


def test_executor_rejects_expired_deadline():
    expired = SkillContext(
        request_id="expired", trace_id="expired",
        deadline_at=datetime.now(timezone.utc) - timedelta(milliseconds=1),
    )
    result = SkillExecutor(get_default_registry()).execute(
        "math.input_guard@1", {"query": "x=1"}, expired
    )
    assert result.status == SkillStatus.FATAL_ERROR
    assert "超时" in result.safe_error


def test_executor_enforces_per_skill_timeout():
    registry = SkillRegistry()
    registry.register(SkillManifest(
        id="math.slow", version="1.0.0", description="Test-only bounded slow skill.",
        input_schema="agentic_rag.domain.schemas.QueryInput",
        output_schema="agentic_rag.domain.schemas.QueryInput",
        handler="tests.skill_runtime.fixtures.slow_handler",
        timeout_ms=5,
    ))
    result = SkillExecutor(registry).execute(
        "math.slow@1", {"query": "x=1"}, context()
    )
    assert result.status == SkillStatus.RETRYABLE_ERROR
    assert "超时" in result.safe_error


def test_all_pipeline_manifests_validate():
    loader = PipelineLoader(get_default_registry())
    paths = sorted((ROOT / "agentic_rag" / "pipelines").glob("*.yaml"))
    assert len(paths) == 6
    assert all(loader.load(path).nodes for path in paths)


def test_pipeline_rejects_unbounded_cycle(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("id: math.bad\nversion: 1.0.0\nsla_ms: 1000\nentry: a\nnodes:\n  a: {skill: math.input_guard@1, next: a}\n", encoding="utf-8")
    with pytest.raises(PipelineError, match="cycle"):
        PipelineLoader(get_default_registry()).load(path)


def test_correction_pipeline_executes_deterministic_path():
    registry = get_default_registry()
    pipeline = PipelineLoader(registry).load(ROOT / "agentic_rag" / "pipelines" / "correction.yaml")
    state = PipelineExecutor(SkillExecutor(registry)).run(pipeline, {"query": "解方程 2x+3=11", "language": "zh"}, context())
    assert state["curriculum_solve"].handled
    assert "x = 4" in state["response_render"].answer
