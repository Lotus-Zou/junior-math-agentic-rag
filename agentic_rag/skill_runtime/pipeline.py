"""YAML pipeline loading, DAG validation, and surface-neutral execution."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_rag.domain.schemas import (
    CriticOutput,
    CurriculumSolveOutput,
    ResponseEnvelope,
    RetrievalOutput,
)
from agentic_rag.skill_runtime.contracts import SkillContext, SkillStatus
from agentic_rag.skill_runtime.errors import PipelineError
from agentic_rag.skill_runtime.executor import SkillExecutor
from agentic_rag.skill_runtime.registry import SkillRegistry
from agentic_rag.skill_runtime.router import SkillRouter

END = "END"


class PipelineNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str = "skill"
    skill: str | None = None
    skills: list[str] = Field(default_factory=list)
    next: str | None = None
    branches: dict[str, str] = Field(default_factory=dict)
    on: dict[str, str] = Field(default_factory=dict)
    input_from: str | None = None

    @model_validator(mode="after")
    def validate_kind(self):
        if self.type == "skill" and not self.skill:
            raise ValueError("skill node requires skill")
        if self.type == "parallel" and not self.skills:
            raise ValueError("parallel node requires skills")
        if self.type == "router" and not self.branches:
            raise ValueError("router node requires branches")
        if self.type not in {"skill", "parallel", "router"}:
            raise ValueError(f"unsupported node type: {self.type}")
        return self


class PipelineManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    version: str
    sla_ms: int = Field(ge=1, le=300000)
    entry: str
    nodes: dict[str, PipelineNode]

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineManifest":
        try:
            return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
        except Exception as exc:
            raise PipelineError(f"Invalid pipeline {path}: {exc}") from exc


class PipelineLoader:
    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def load(self, path: str | Path) -> PipelineManifest:
        pipeline = PipelineManifest.from_yaml(path)
        self.validate(pipeline)
        return pipeline

    def validate(self, pipeline: PipelineManifest) -> None:
        if pipeline.entry not in pipeline.nodes:
            raise PipelineError(f"Missing entry node: {pipeline.entry}")
        edges: dict[str, list[str]] = {}
        timeout_total = 0
        for name, node in pipeline.nodes.items():
            targets = [item for item in [node.next, *node.branches.values(), *node.on.values()] if item and item != END]
            missing = [target for target in targets if target not in pipeline.nodes]
            if missing:
                raise PipelineError(f"Node {name} points to missing nodes: {missing}")
            edges[name] = targets
            refs = [node.skill] if node.skill else node.skills
            manifests = [self.registry.resolve(ref) for ref in refs]
            timeout_total += max((item.timeout_ms for item in manifests), default=0)
            if node.type == "parallel":
                effects = [effect for item in manifests for effect in item.side_effects]
                if len(effects) != len(set(effects)):
                    raise PipelineError(f"Parallel node {name} has conflicting side effects")
        if timeout_total > pipeline.sla_ms * 3:
            raise PipelineError("Declared Skill timeouts cannot fit bounded pipeline retries")
        visited: set[str] = set()
        active: set[str] = set()

        def visit(name: str):
            if name in active:
                raise PipelineError(f"Unbounded cycle detected at {name}")
            if name in visited:
                return
            active.add(name)
            for target in edges[name]:
                visit(target)
            active.remove(name)
            visited.add(name)

        visit(pipeline.entry)
        if visited != set(pipeline.nodes):
            raise PipelineError(f"Unreachable nodes: {sorted(set(pipeline.nodes) - visited)}")


class PipelineExecutor:
    def __init__(self, skill_executor: SkillExecutor, *, router: SkillRouter | None = None):
        self.skills = skill_executor
        self.router = router or SkillRouter()

    def run(self, pipeline: PipelineManifest, payload: dict[str, Any], context: SkillContext) -> dict[str, Any]:
        state: dict[str, Any] = {"input": payload, **payload}
        current = pipeline.entry
        while current != END:
            node = pipeline.nodes[current]
            if node.type == "router":
                current = self.router.choose(state, node.branches)
                continue
            if node.type == "parallel":
                with ThreadPoolExecutor(max_workers=len(node.skills)) as pool:
                    futures = [
                        pool.submit(
                            self.skills.execute, ref,
                            self._project_input(ref, state, payload, node.input_from),
                            context, pipeline=pipeline.ref,
                        )
                        for ref in node.skills
                    ]
                    results = [future.result() for future in futures]
                state[current] = [result.value for result in results if result.status == SkillStatus.OK]
                if not state[current] and results:
                    state["safe_error"] = results[0].safe_error
                    state["failure"] = {
                        "node": current,
                        "skills": list(node.skills),
                        "status": results[0].status.value,
                        "safe_error": results[0].safe_error,
                    }
            else:
                ref = node.skill or ""
                node_payload = self._project_input(ref, state, payload, node.input_from)
                result = self.skills.execute(ref, node_payload, context, pipeline=pipeline.ref)
                state[current] = result.value
                state["last_result"] = result
                if result.status != SkillStatus.OK:
                    state["safe_error"] = result.safe_error
                    state["failure"] = {
                        "node": current,
                        "skill": ref,
                        "status": result.status.value,
                        "safe_error": result.safe_error,
                    }
                if node.on:
                    key = self.branch_key(result.value, result.status)
                    current = node.on.get(key, node.on.get("fail", END))
                    continue
            current = node.next or END
        return state

    @staticmethod
    def branch_key(value: Any, status: SkillStatus = SkillStatus.OK) -> str:
        if status != SkillStatus.OK:
            return "retryable_fail" if status == SkillStatus.RETRYABLE_ERROR else "fail"
        if isinstance(value, CriticOutput):
            return "pass" if value.passed else "fail"
        if isinstance(value, RetrievalOutput):
            return "pass" if value.candidates else "fail"
        return "pass"

    def _project_input(self, ref: str, state: dict[str, Any], base: dict[str, Any], input_from: str | None) -> dict[str, Any]:
        manifest = self.skills.registry.resolve(ref)
        merged: dict[str, Any] = dict(base)
        selected = state.get(input_from) if input_from else None
        values = [selected] if selected is not None else list(state.values())
        for value in values:
            if hasattr(value, "model_dump"):
                merged.update(value.model_dump())
            elif isinstance(value, dict):
                merged.update(value)
            elif isinstance(value, list) and value and all(hasattr(item, "model_dump") for item in value):
                dumps = [item.model_dump() for item in value]
                if all("candidates" in item for item in dumps):
                    merged["rankings"] = [item["candidates"] for item in dumps]
        latest_response_applied = False
        for value in reversed(list(state.values())):
            if not hasattr(value, "model_dump"):
                continue
            dump = value.model_dump()
            if (
                not latest_response_applied
                and "response" in dump
                and isinstance(dump["response"], dict)
                and dump["response"]
            ):
                merged.update(dump["response"])
                latest_response_applied = True
            if "candidates" in dump and "candidates" not in merged:
                merged["candidates"] = dump["candidates"]
                merged["contexts"] = dump["candidates"]
            if "answer" in dump and "answer" not in merged:
                merged["answer"] = dump["answer"]
        if "candidates" in merged:
            merged["contexts"] = merged["candidates"]
        turn = state.get("turn_router")
        if turn is not None and hasattr(turn, "routed_query") and not ref.startswith("math.curriculum_solve"):
            merged["query"] = turn.routed_query
        if (
            ref.startswith("math.answer_generate")
            or ref.startswith("math.curriculum_tutor")
            or ref.startswith("math.exercise_generate")
        ) and turn is not None:
            merged["intent"] = turn.intent
        if ref.startswith("math.response_render"):
            for value in reversed(list(state.values())):
                if not hasattr(value, "candidates"):
                    continue
                merged["sources"] = [item.model_dump() for item in value.candidates]
                break
            raw_metrics = (
                merged.get("metrics", {})
                if isinstance(merged.get("metrics"), dict)
                else {}
            )
            prior_metrics = {
                key: raw_metrics[key]
                for key in (
                    "tool_calls",
                    "model_attempts",
                    "model_successes",
                    "model_failures",
                    "latency_ms",
                )
                if key in raw_metrics
            }
            attempts = int(prior_metrics.get("model_attempts") or 0)
            successes = int(
                prior_metrics.get("model_successes")
                or prior_metrics.get("tool_calls")
                or 0
            )
            failures = int(prior_metrics.get("model_failures") or 0)
            for node_name in (
                "rerank_filter",
                "web_search",
                "answer_generate",
                "answer_critic",
                "answer_repair",
                "answer_repair_critic",
            ):
                value = state.get(node_name)
                if value is None:
                    continue
                attempts += int(getattr(value, "model_attempts", 0) or 0)
                successes += int(getattr(value, "model_successes", 0) or 0)
                failures += int(getattr(value, "model_failures", 0) or 0)
            merged["metrics"] = {
                **prior_metrics,
                "tool_calls": successes,
                "model_attempts": attempts,
                "model_successes": successes,
                "model_failures": failures,
            }
        render_contract = self._trusted_render_contract(state) if ref.startswith("math.response_render") else None
        if render_contract is not None:
            merged.update(render_contract)
        fields = manifest.input_model.model_fields
        projected = {name: merged[name] for name in fields if name in merged}
        if ref.startswith("math.answer_critic") and projected.get("contexts"):
            projected["contexts"] = [
                item.get("content", "") if isinstance(item, dict) else str(item)
                for item in projected["contexts"]
            ]
        if ref.startswith("math.response_render") and projected.get("sources"):
            projected["sources"] = [
                {
                    "chunk_id": item.get("chunk_id"),
                    "source": item.get("source", ""),
                    "chapter": item.get("chapter") or item.get("metadata", {}).get("chapter"),
                    "rank": item.get("rank") or item.get("metadata", {}).get("rank"),
                    "excerpt": str(item.get("content", ""))[:1200] or None,
                }
                if isinstance(item, dict)
                else item
                for item in projected["sources"]
            ]
        if "query" in fields and "query" not in projected and merged.get("rewritten_query"):
            projected["query"] = merged["rewritten_query"]
        return projected

    @staticmethod
    def _trusted_render_contract(state: dict[str, Any]) -> dict[str, Any] | None:
        for value in reversed(list(state.values())):
            if isinstance(value, CurriculumSolveOutput) and value.response is not None:
                return {
                    "response_type": value.response.response_type,
                    "validation_passed": value.response.validation_passed,
                }
            if isinstance(value, ResponseEnvelope):
                return {
                    "response_type": value.response_type,
                    "validation_passed": value.validation_passed,
                }
            if isinstance(value, CriticOutput):
                return {
                    "response_type": (
                        "verified_answer" if value.passed else "clarification_required"
                    ),
                    "validation_passed": value.passed,
                }
        return None
