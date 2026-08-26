"""Single execution boundary for validation, policy, retries, timeouts, and traces."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextvars import copy_context
from typing import Any

from pydantic import ValidationError

from agentic_rag.skill_runtime.contracts import SkillContext, SkillResult, SkillStatus
from agentic_rag.skill_runtime.errors import SkillRuntimeError
from agentic_rag.skill_runtime.policies import PolicyEngine
from agentic_rag.skill_runtime.registry import SkillRegistry
from agentic_rag.skill_runtime.telemetry import TelemetrySink, TraceEvent


class SkillExecutor:
    def __init__(self, registry: SkillRegistry, *, telemetry: TelemetrySink | None = None, policies: PolicyEngine | None = None):
        self.registry = registry
        self.telemetry = telemetry or TelemetrySink()
        self.policies = policies or PolicyEngine()

    def execute(self, ref: str, payload: Any, context: SkillContext, *, pipeline: str = "") -> SkillResult[Any]:
        manifest = self.registry.resolve(ref)
        started = time.perf_counter()
        status = SkillStatus.FATAL_ERROR
        artifacts = []
        decisions: list[str] = []
        safe_error = ""
        result_value = None
        try:
            if context.remaining_budget_ms <= 0:
                raise SkillRuntimeError("Request deadline exceeded", safe_message="本次处理已超时，请重试。")
            validated = manifest.input_model.model_validate(payload)
            decisions = self.policies.authorize(manifest, context)
            timeout_ms = min(manifest.timeout_ms, context.remaining_budget_ms)
            last_error: Exception | None = None
            for _ in range(manifest.max_attempts):
                try:
                    pool = ThreadPoolExecutor(max_workers=1)
                    execution_context = copy_context()
                    future = pool.submit(execution_context.run, manifest.handler_callable, validated, context)
                    try:
                        raw = future.result(timeout=timeout_ms / 1000)
                    finally:
                        pool.shutdown(wait=False, cancel_futures=True)
                    result_value = manifest.output_model.model_validate(raw)
                    status = SkillStatus.OK
                    last_error = None
                    break
                except FutureTimeout as exc:
                    last_error = exc
                except SkillRuntimeError as exc:
                    last_error = exc
                    if not exc.retryable:
                        break
            if last_error:
                raise last_error
        except ValidationError:
            safe_error = "请求参数格式不正确，请检查后重试。"
        except SkillRuntimeError as exc:
            status = SkillStatus.RETRYABLE_ERROR if exc.retryable else SkillStatus.FATAL_ERROR
            safe_error = exc.safe_message
        except FutureTimeout:
            status = SkillStatus.RETRYABLE_ERROR
            safe_error = "处理超时，请稍后重试。"
        except Exception:
            safe_error = "系统暂时无法处理该请求，请稍后重试。"

        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        event = TraceEvent(
            trace_id=context.trace_id,
            pipeline=pipeline,
            skill=manifest.ref,
            status=status.value,
            latency_ms=latency_ms,
            input_hash=self.telemetry.hash_input(payload),
            artifact_ids=[item.artifact_id for item in artifacts],
            policy_decisions=decisions,
        )
        self.telemetry.emit(event)
        return SkillResult(
            status=status,
            value=result_value,
            artifacts=artifacts,
            metrics={"latency_ms": latency_ms, "attempt_limit": manifest.max_attempts},
            provenance={"skill": manifest.id, "version": manifest.version},
            safe_error=safe_error,
        )
