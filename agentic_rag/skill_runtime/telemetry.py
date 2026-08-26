"""Structured per-Skill telemetry with hashes instead of raw student input."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import BaseModel, ConfigDict


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trace_id: str
    pipeline: str = ""
    skill: str
    status: str
    latency_ms: float
    cache: str = "miss"
    input_hash: str
    artifact_ids: list[str]
    policy_decisions: list[str]


class TelemetrySink:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.events: list[TraceEvent] = []
        self._lock = Lock()

    @staticmethod
    def hash_input(payload: Any) -> str:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def emit(self, event: TraceEvent) -> None:
        with self._lock:
            self.events.append(event)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(event.model_dump_json() + "\n")

