"""Shared adapter context and schema helpers."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from pydantic import create_model

from agentic_rag.skill_runtime.contracts import SkillContext


def adapter_context(language: str = "zh", timeout_ms: int = 8000) -> SkillContext:
    identifier = str(uuid.uuid4())
    return SkillContext(
        request_id=identifier, trace_id=identifier, language=language,
        deadline_at=datetime.now(timezone.utc) + timedelta(milliseconds=timeout_ms),
    )


def wrapped_input_model(manifest):
    return create_model(f"{manifest.id.replace('.', '_')}_arguments", payload=(manifest.input_model, ...))
