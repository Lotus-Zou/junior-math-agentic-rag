"""Thread-safe progress and token delivery for request-scoped SSE streams."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock

_lock = Lock()
_sinks: dict[str, Callable[[dict], None]] = {}


def register(trace_id: str, sink: Callable[[dict], None]) -> None:
    with _lock:
        _sinks[trace_id] = sink


def unregister(trace_id: str) -> None:
    with _lock:
        _sinks.pop(trace_id, None)


def emit(trace_id: str, event: dict) -> None:
    with _lock:
        sink = _sinks.get(trace_id)
    if sink is None:
        return
    try:
        sink(event)
    except Exception:
        # Streaming is observational; a disconnected browser must never fail the answer.
        return
